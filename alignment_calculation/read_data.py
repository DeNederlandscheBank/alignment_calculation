from typing import Union
import pandas as pd
from vl_connect import devo


def _add_orbis_lei_based(
    pacta_data: pd.DataFrame, lei_column: str, external_columns: dict
) -> pd.DataFrame:
    """
    Adds data from orbis based on lei code

    Parameters:
    -----------

    data: pandas.DataFrame
        A dataframe with external_ids and possible company_lei and parent_lei
        columns to which the orbis data should be added
    lei_column: str
        The name of the column with the lei codes in pacta_data
    extra_columns : dict
        The extra columns for which data should be fetched. The value relates to the 
        expression that should be added to the query to get the additional data and
        the key relates to how the additional datapoint should be called.

    Returns:
    --------
    pandas.DataFrame
        The pacta_data dataframe with the additional orbis data.
    """

    pacta_unique = pacta_data.drop_duplicates(subset=[lei_column]).dropna(
        subset=[lei_column]
    )
    orbis_data_lei = _add_orbis_data(
        leis=str(pacta_unique[lei_column].tolist())[1:-1],
        extra_columns=external_columns,
    ).drop(columns=["external_id"])
    orbis_data_lei = orbis_data_lei.merge(
        pacta_unique[[lei_column, "external_id"]],
        left_on=["lei"],
        right_on=[lei_column],
    ).drop(columns=["lei", lei_column])

    return orbis_data_lei


def _add_external_data(
    pacta_data: pd.DataFrame, external_columns: dict
) -> pd.DataFrame:
    """
    Adds data from orbis based on bvdidnumber

    Parameters:
    -----------

    data: pandas.DataFrame
        A dataframe with external_ids and possible company_lei and parent_lei
        columns to which the orbis data should be added
    extra_columns : dict
        The extra columns for which data should be fetched. The value relates to the 
        expression that should be added to the query to get the additional data and
        the key relates to how the additional datapoint should be called.

    Returns:
    --------
    pandas.DataFrame
        The processed dataframe with the additional orbis data.
    """
    pacta_unique = pacta_data.drop_duplicates(subset=["external_id"]).dropna(
        subset=["external_id"]
    )
    num_splits = len(pacta_unique) // 7000 + 1
    orbis_data = []

    for i in range(num_splits):
        start_idx = i * 7000
        end_idx = min((i + 1) * 7000, len(pacta_unique))
        orbis_data.append(
            _add_orbis_data(
                bvdids=str(pacta_unique[start_idx:end_idx]["external_id"].tolist())[
                    1:-1
                ],
                extra_columns=external_columns,
            ).drop(columns=["lei"])
        )

    if "company_lei" in pacta_data.columns:
        orbis_data.append(
            _add_orbis_lei_based(pacta_data, "company_lei", external_columns)
        )

    if "parent_lei" in pacta_data.columns:
        orbis_data.append(
            _add_orbis_lei_based(pacta_data, "parent_lei", external_columns)
        )

    orbis_data = (
        pd.concat(orbis_data)
        .groupby(["external_id"], as_index=False)
        .max(numeric_only=True)
    )

    return pacta_data.merge(
        orbis_data, how="left", left_on=["external_id"], right_on=["external_id"]
    )


def _load_loan_data(
    pacta_data: pd.DataFrame,
    year: int,
    month: int,
    portfolio_codes: list,
    start_month: int,
    start_year: int,
    frequency: str,
    additional_columns: list,
) -> pd.DataFrame:
    """
    Load loan data based on the counterparty_id for the combined PACTA data.

    Parameters:
    -----------
    pacta_data : pd.DataFrame
        DataFrame containing the combined PACTA data.

    year : int
        The year for loan data.

    month : int
        The month for loan data.

    portfolio_codes : list
        List of portfolio codes for filtering loan data.

    start_month : int
        The start month for loan data.

    start_year : int
        The start year for loan data.

    frequency : str
        The frequency for which the data should be loaded.

    additional_columns : list
        Additional columns to include in loan data.

    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame containing AnaCredit loan data.
    """
    pacta_unique = pacta_data.drop_duplicates(subset=["counterparty_id"])
    num_splits = len(pacta_unique) // 7000 + 1
    anacredit_data = []

    for i in range(num_splits):
        start_idx = i * 7000
        end_idx = min((i + 1) * 7000, len(pacta_unique))
        anacredit_data.append(
            _load_anacredit_loan_data(
                year=year,
                month=month,
                portfolio_codes=portfolio_codes,
                start_month=start_month,
                start_year=start_year,
                additional_columns=additional_columns,
                frequency=frequency,
                riad_ids=str(
                    pacta_unique[start_idx:end_idx]["counterparty_id"].tolist()
                )[1:-1],
            )
        )
    return pd.concat(anacredit_data)


def _load_loan_counterparties(
    year: int, month: int, start_year: int, start_month: int, include_orbis: bool = True
) -> pd.DataFrame:
    """
    Reads the counterparty data from AnaCredit by using DEVO

    Parameters
    ----------
    year : int
        The year for which the data should be fetched as well as the year
        for which the corporate structure of the banks should be determined.
    month : int
        The month for which the data should be fetched.
    start_year : int
        The start_year if all the data till the present day should be fetched.
    start_month : int
        The start_month if all the data till the present day should be fetched.
    include_orbis : bool, optional
        flag to use the orbis data for the determination of the relevant
        counterparties.
        default=True

    Returns
    -------
    pandas.DataFrame
        The AnaCredit counterparty data as fetch according to the conditions
    """

    single_period = False
    if start_year is None:
        single_period = True
        start_year = year
    if start_month is None:
        start_month = month

    bvd_major_sector = [
        "Metals & Metal Products",
        "Mining & Extraction",
        "Transport Manufacturing",
        "Utilities",
        "Biotechnology and Life Sciences",
    ]
    naics_major_sector = [
        "Petroleum and Coal Products Manufacturing",
        "Cement and Concrete Product Manufacturing",
        "Metal and Mineral (except Petroleum) Merchant Wholesalers",
        "Electric Power Generation, Transmission and Distribution",
        "Lime and Gypsum Product Manufacturing",
    ]
    naics_non_sector = [
        "Monetary Authorities-Central Bank",
        "Household Appliances and Electrical and Electronic Goods Merchant Wholesalers",
        "Manufacturing and Reproducing Magnetic and Optical Media",
        "Depository Credit Intermediation",
        "Glass and Glass Product Manufacturing",
        "Plastics Product Manufacturing",
        "Other Heavy and Civil Engineering Construction",
        "Wired and Wireless Telecommunications (except Satellite)",
        "Insurance Carriers",
        "Pesticide, Fertilizer, and Other Agricultural Chemical Manufacturing",
        "Agriculture, Construction, and Mining Machinery Manufacturing",
        "Chemical and Allied Products Merchant Wholesalers",
        "Other Nonmetallic Mineral Product Manufacturing",
        "Pharmaceutical and Medicine Manufacturing",
        "Clay Product and Refractory Manufacturing",
        "Grocery and Related Product Merchant Wholesalers",
        "Other General Purpose Machinery Manufacturing",
        "Automotive Repair and Maintenance",
        "Services to Buildings and Dwellings",
        "Beer, Wine, and Distilled Alcoholic Beverage Merchant Wholesalers",
        "Rubber Product Manufacturing",
        "Agencies, Brokerages, and Other Insurance Related Activities",
        "Building Finishing Contractors",
        "Chemical Manufacturing",
        "Real Estate",
        "Professional and Commercial Equipment and Supplies Merchant Wholesalers",
        "Ventilation, Heating, Air-Conditioning, and Commercial Refrigeration Equipment Manufacturing",
        "Securities and Commodity Exchanges",
        "Medical Equipment and Supplies Manufacturing",
        "Motor Vehicle Body and Trailer Manufacturing",
        "Paper and Paper Product Merchant Wholesalers",
        "Paint, Coating, and Adhesive Manufacturing",
        "Other Support Services",
        "Electric Lighting Equipment Manufacturing",
        "Electronic and Precision Equipment Repair and Maintenance",
        "Facilities Support Services",
        "Fruit and Vegetable Preserving and Specialty Food Manufacturing",
        "Furniture and Home Furnishing Merchant Wholesalers",
        "Grain and Oilseed Milling",
        "Household Appliance Manufacturing",
        "Nonresidential Building Construction",
        "Justice, Public Order, and Safety Activities",
        "Lessors of Nonfinancial Intangible Assets (except Copyrighted Works)",
        "Drugs and Druggists' Sundries Merchant Wholesalers",
        "Legal Services",
    ]
    bvd_non_sector = [
        "Agriculture, Horticulture & Livestock",
        "Food & Tobacco Manufacturing",
        "Retail",
        "Textiles & Clothing Manufacturing",
        "Information Services",
        "Waste Management & Treatment",
        "Printing & Publishing",
        "Computer Hardware",
        "Computer Software",
        "Media & Broadcasting",
        "Property Services",
        "Travel, Personal & Leisure",
        "Transport, Freight & Storage",
        "Wood, Furniture & Paper Manufacturing",
    ]
    nace_non_sector = [
        "S - Other service activities",
        "P - Education",
        "Q - Human health and social work activities",
        "E - Water supply; sewerage, waste management and remediation activities",
        "L - Real estate activities",
        "O - Public administration and defence; compulsory social security",
    ]
    sub_nace_non_sector = [
        "Regulation of and contribution to more efficient operation of businesses",
        "Manufacture of instruments and appliances for measuring, testing and navigation",
        "Other construction installation",
        "Other building completion and finishing",
        "Architectural activities",
        "Wholesale of hardware, plumbing and heating equipment and supplies",
        "Manufacture of other inorganic basic chemicals",
        "Manufacture of industrial gases",
        "Construction of utility projects for electricity and telecommunications",
        "Other financial service activities, except insurance and pension funding nec",
        "Non-specialised wholesale trade",
        "Manufacture of soap and detergents, cleaning and polishing preparations",
        "Electrical installation",
        "Manufacture of man-made fibres",
        "Other business support service activities nec",
        "Engineering activities and related technical consultancy",
        "Manufacture of electricity distribution and control apparatus",
        "Manufacture of batteries and accumulators",
        "Agents involved in the sale of a variety of goods",
        "Other credit granting",
        "Construction of roads and motorways",
        "Manufacture of communication equipment",
        "Manufacture of electronic components",
        "Administration of financial markets",
        "Wholesale of wood, construction materials and sanitary equipment",
        "Business and other management consultancy activities",
        "Manufacture of plastics in primary forms",
        "Security and commodity contracts brokerage",
        "Plumbing, heat and air conditioning installation",
        "Wholesale of grain, unmanufactured tobacco, seeds and animal feeds",
    ]
    naics_filter_two = [
        "Foreign trade and international banking institutions",
        "Functions related to depository banking, not elsewhere classified",
        "Mortgage bankers loan correspondents",
        "Offices of bank holding companies",
        "Federal and federally-sponsored credit agencies",
        "Miscellaneous business credit institutions",
        "Pension, health, and welfare funds",
        "Provincial public administration",
    ]

    orbis_table = f"""WITH  
        orbis_eligible AS(
            SELECT DISTINCT
                map_orbis.entty_riad_cd AS entty_riad_cd,
                map_orbis.bvd_id AS bvdidnumber
            FROM lab_prj_lab_riad.riad_orbis_mppng_d_1 AS map_orbis
            INNER JOIN crp_orbis.orbis_industry_classifications AS orbis_sector
                ON orbis_sector.bvdidnumber = map_orbis.bvd_id
            WHERE (map_orbis.mtchng_mthd = 'ID based'
                    OR map_orbis.smlrty_scr > 99)
                AND ((orbis_sector.bvdmajorsector IN ({str(bvd_major_sector)[1:-1]}) OR 
                    orbis_sector.naics2012corecodetextdescription IN ({str(naics_major_sector)[1:-1]})) OR 
                    (orbis_sector.bvdmajorsector NOT IN ({str(bvd_non_sector)[1:-1]}) AND
                        orbis_sector.nacerev2mainsection NOT IN ({str(nace_non_sector)[1:-1]}) AND
                        orbis_sector.naics2012corecodetextdescription NOT IN ({str(naics_non_sector)[1:-1]}) AND
                        orbis_sector.nacerev2corecodetextdescription NOT IN ({str(sub_nace_non_sector)[1:-1]}) AND
                        orbis_sector.ussicprimarycodetextdescription NOT IN ({str(naics_filter_two)[1:-1]})))

            UNION

            SELECT DISTINCT
                map_orbis.entty_riad_cd AS entty_riad_cd,
                map_orbis.bvd_id AS bvdidnumber
            FROM lab_prj_lab_riad.riad_orbis_mppng_d_1 AS map_orbis
            LEFT ANTI JOIN crp_orbis.orbis_industry_classifications AS orbis_sector
                ON orbis_sector.bvdidnumber = map_orbis.bvd_id
            WHERE (map_orbis.mtchng_mthd = 'ID based'
                    OR map_orbis.smlrty_scr > 99)
        ),   
        orbis_parents AS (
            SELECT DISTINCT
                map_orbis.entty_riad_cd AS entty_riad_cd,
                links.shareholderbvdid AS bvdidnumber
            FROM lab_prj_lab_riad.riad_orbis_mppng_d_1 AS map_orbis
            INNER JOIN crp_orbis.orbis_links_current AS links
                ON links.subsidiarybvdid = map_orbis.bvd_id            
            WHERE (map_orbis.mtchng_mthd = 'ID based'
                    OR map_orbis.smlrty_scr > 99)
                AND links.typeofrelation = 'GUO 50'
                AND links.shareholderbvdid IN (SELECT bvdidnumber FROM orbis_eligible)
                AND links.subsidiarybvdid <> links.shareholderbvdid
            ),
        orbis_combi AS (
            SELECT DISTINCT
                orbis_name.name AS name,
                orbis_id.leilegalentityidentifier AS leilegalentityidentifier,
                orbis_eligible.entty_riad_cd AS entty_riad_cd,
                orbis_eligible.bvdidnumber AS bvdidnumber
            FROM orbis_eligible AS orbis_eligible
            LEFT JOIN crp_orbis.orbis_bvd_id_and_name AS orbis_name
                ON orbis_eligible.bvdidnumber = orbis_name.bvdidnumber
            LEFT JOIN crp_orbis.orbis_identifiers AS orbis_id
                ON orbis_eligible.bvdidnumber = orbis_id.bvdidnumber

            UNION

            SELECT DISTINCT
                orbis_name.name AS name,
                orbis_id.leilegalentityidentifier AS leilegalentityidentifier,
                orbis_parents.entty_riad_cd AS entty_riad_cd,
                orbis_parents.bvdidnumber AS bvdidnumber
            FROM orbis_parents AS orbis_parents
            LEFT JOIN crp_orbis.orbis_identifiers AS orbis_id
                ON orbis_parents.bvdidnumber = orbis_id.bvdidnumber
            LEFT JOIN crp_orbis.orbis_bvd_id_and_name AS orbis_name
                ON orbis_parents.bvdidnumber = orbis_name.bvdidnumber

            UNION
            SELECT DISTINCT
                dbtr.nm_entty_le AS name,
                dbtr.lei AS leilegalentityidentifier,
                dbtr.entty_riad_cd AS entty_riad_cd,
                '-' AS bvdidnumber
            FROM crp_anacredit.anacredit_dm_d_dbtr_cc AS dbtr
            LEFT ANTI JOIN lab_prj_lab_riad.riad_orbis_mppng_d_1 AS map_orbis
                ON map_orbis.entty_riad_cd = dbtr.entty_riad_cd
            WHERE
                dbtr.dt_rfrnc {'=' if single_period else '>='} {start_year}{f'{start_month : 03d}'.strip()}
                AND RIGHT(CAST(dbtr.dt_rfrnc AS STRING), 2) IN ('03','06','09','12')
                AND ((dbtr.instttnl_sctr = 'S11'
                        OR dbtr.instttnl_sctr_le = 'S11') OR 
                    dbtr.ecnmc_actvty IN ('5', '5_1', '5_10', '5_2', '5_20', '6', '6_1', '6_10', '6_2', '6_20', 
                    '23', '23_5', '23_51', '24', '24_1', '24_10', '29', '29_1', '29_10', '35', '35_1', '35_11'))
        )
        """

    query = f"""{orbis_table if include_orbis else ''}
            SELECT
                dbtr.entty_riad_id AS counterparty_id,
                dbtr.nm_entty_le AS company_name,
                MAX(dbtr.cntry) AS company_country,
                dbtr.lei AS company_lei
                {', MAX(orbis_eligible.bvdidnumber) AS external_id' if include_orbis else ''}
                {', orbis_eligible.name AS parent_name' if include_orbis else ''}
                {', orbis_eligible.leilegalentityidentifier AS parent_lei' if include_orbis else ''}
            FROM crp_anacredit.anacredit_dm_d_dbtr_cc AS dbtr
            {'INNER JOIN orbis_combi AS orbis_eligible' if include_orbis else ''}
                {'ON dbtr.entty_riad_cd = orbis_eligible.entty_riad_cd' if include_orbis else ''}
            WHERE
                dbtr.dt_rfrnc {'=' if single_period else '>='} {start_year}{f'{start_month : 03d}'.strip()}
                AND RIGHT(CAST(dbtr.dt_rfrnc AS STRING), 2) IN ('03','06','09','12')
                AND ((dbtr.instttnl_sctr = 'S11'
                        OR dbtr.instttnl_sctr_le = 'S11') OR 
                    dbtr.ecnmc_actvty IN ('5', '5_1', '5_10', '5_2', '5_20', '6', '6_1', '6_10', '6_2', '6_20', 
                    '23', '23_5', '23_51', '24', '24_1', '24_10', '29', '29_1', '29_10', '35', '35_1', '35_11'))
            GROUP BY
                dbtr.entty_riad_id,
                dbtr.lei,
                dbtr.nm_entty_le
                {', orbis_eligible.leilegalentityidentifier' if include_orbis else ''}
                {', orbis_eligible.name' if include_orbis else ''}
            """

    return devo.read_sql(query)


def _add_orbis_data(
    bvdids: str = "''", leis: str = "''", extra_columns: dict = None
) -> pd.DataFrame:
    """
    Reads orbis total_assets, turnover and currentratio data through the use of DEVO

    Parameters
    ----------
    bvdids : str
        The bvdids for which the additional data is to be fetched
        default=  "''"
    leis : str
        The leis for which the additional data is to be fetched
        default=  "''"
    extra_columns : dict
        The extra columns for which data should be fetched. The value relates to the
        expression that should be added to the query to get the additional data and
        the key relates to how the additional datapoint should be called.
        default=  None


    Returns
    -------
    pandas.DataFrame
        The Orbis data as fetch according to the conditions
    """
    if extra_columns is None:
        extra_columns = {}

    query = f"""
        WITH cte AS(
            SELECT 
                fin.bvdidnumber,
                MAX(ids.leilegalentityidentifier) AS lei,
                MAX(CAST(LEFT(fin.closingdate, 4) AS INT)) AS max_year
            FROM 
                crp_orbis.orbis_key_financials_eur AS fin
            INNER JOIN
                crp_orbis.orbis_identifiers AS ids
                ON ids.bvdidnumber = fin.bvdidnumber
            WHERE 
                operatingrevenueturnover IS NOT NULL
                AND (fin.bvdidnumber IN ({bvdids}) OR
                     ids.leilegalentityidentifier IN ({leis}) 
                    )
            GROUP BY 
                fin.bvdidnumber
            )

        SELECT 
            fin.bvdidnumber AS external_id,
            {'' if len(extra_columns)==0 else (', '.join(
                [f'{value} AS {key}' for key, value in extra_columns.items()]) + ',')}
            MAX(cte.lei) AS lei
        FROM 
            crp_orbis.orbis_key_financials_eur AS fin
        INNER JOIN 
            cte
            ON fin.bvdidnumber = cte.bvdidnumber
            AND CAST(LEFT(fin.closingdate, 4) AS INT) = cte.max_year
        GROUP BY
            fin.bvdidnumber;
        """

    return devo.read_sql(query)


def _load_anacredit_loan_data(
    year: int,
    month: int,
    portfolio_codes: list,
    start_year: int,
    start_month: int,
    frequency: str,
    riad_ids: str,
    additional_columns: dict,
) -> pd.DataFrame:
    """
    Reads the AnaCredit loan data through the use of DEVO

    Parameters
    ----------
    year : int
        The year for which the data should be fetched as well as the year
        for which the corporate structure of the banks should be determined.
    month : int
        The month for which the data should be fetched.
    portfolio_codes : list
        A list of portfolio codes to which the data collecting should be limited.
    start_year : int
        The start_year if all the data till the present day should be fetched.
    start_month : int
        The start_month if all the data till the present day should be fetched.
    frequency: str
        String indicating what the frequency of the fetched data should be,
        - 'M': Monthly
        - 'Q': Quarterly
        - 'Y': Yearly
    riad_ids : str
        The valid riad_ids as a string compatible with an SQL WHERE IN clasue
        of the counterparties that should be considered when fetching the data.
    additional_columns : dict
        Any additional columns that should be fetched from the AnaCredit data.

    Returns
    -------
    pandas.DataFrame
        The AnaCredit data as fetch according to the conditions
    """

    single_period = False
    if start_year is None:
        single_period = True
        start_year = year
    if start_month is None:
        start_month = month

    data_frequency_sql = None
    if frequency == "M":
        data_frequency_sql = ""
    elif frequency == "Q":
        data_frequency_sql = (
            "AND RIGHT(CAST(inst.dt_rfrnc AS STRING), 2) IN ('03','06','09','12')"
        )
    elif frequency == "Y":
        data_frequency_sql = "AND RIGHT(CAST(inst.dt_rfrnc AS STRING), 2) = '12'"
    else:
        raise ValueError(
            "The selected frequency should either be 'M', 'Q' or 'Y', please try again with a correct frequency"
        )

    query = f"""
            WITH 
            business_models AS (
            SELECT 
                    imas_jst_code_today AS jst_code,
                    MAX(imas_business_model_today) AS business_model
                FROM 
                    crp_agora.agora_entity_master_data_calendar
                WHERE 
                    imas_business_model_today IS NOT NULL
                    AND imas_jst_code_today <> 'LSIXX'
                    AND imas_is_group_head = 'Y'
                GROUP BY 
                    imas_jst_code_today
            ),banks AS(
            SELECT DISTINCT
                COALESCE(agora.riad_code, agora.imas_riad_code, agora.riad_riad_code) AS riad_code,
                COALESCE(agora.imas_jst_code_today, agora.imas_jst_code) AS jst_code,
                business_models.business_model AS business_model
            FROM 
                crp_agora.agora_entity_master_data_calendar AS agora
            LEFT JOIN
                business_models AS business_models
            ON
                business_models.jst_code = COALESCE(agora.imas_jst_code_today, agora.imas_jst_code)
            WHERE
                agora.reference_period = '{year}-{f'{month : 03d}'.strip()}-{pd.Period(f'{year}-{month}').days_in_month}'
                AND agora.riad_code IS NOT NULL
                AND COALESCE(agora.imas_jst_code_today, agora.imas_jst_code) <> 'LSIXX' 
                AND COALESCE(agora.imas_jst_code_today, agora.imas_jst_code) IS NOT NULL
                {'' if len(portfolio_codes)>0 else '--'} AND COALESCE(agora.imas_jst_code_today, agora.imas_jst_code) IN ({portfolio_codes})
            ) 

            SELECT 
                inst.dbtr_id AS counterparty_id,
                banks.jst_code AS portfolio_code,
                crdt.dt_rfrnc AS portfolio_date,
                AVG(inst.otstndng_nmnl_amnt_cv) AS outstanding_amount,
                {'' if len(additional_columns)==0 else (', '.join(
                    [f'{value} AS {key}' for key, value in additional_columns.items()]) + ',')}
                CONCAT(inst.instrmnt_id, inst.cntrct_id) AS loan_id
            FROM crp_anacredit.anacredit_dm_instrmnt_fct_cc AS inst
            INNER JOIN crp_anacredit.anacredit_dm_d_crdtr_cc AS crdt
                ON inst.dt_rfrnc = crdt.dt_rfrnc
                AND inst.crdtr_id = crdt.entty_riad_id
            INNER JOIN banks AS banks
                ON banks.riad_code = crdt.entty_riad_cd_le
            INNER JOIN crp_anacredit.anacredit_dm_d_dbtr_cc AS dbtr
                ON dbtr.dt_rfrnc = inst.dt_rfrnc
                AND dbtr.entty_riad_id = inst.dbtr_id
            WHERE
                inst.rsdl_mtrty > 0
                AND inst.dt_rfrnc {'=' if single_period else '>='} {start_year}{f'{start_month : 03d}'.strip()}
                {data_frequency_sql}
                AND inst.is_intrcmpny = 'F'
                AND (dbtr.entty_riad_id IN ({riad_ids}) OR
                    dbtr.ultmt_prnt_undrt_id IN ({riad_ids}))
                AND (inst.otstndng_nmnl_amnt_cv > 0 OR
                    inst.off_blnc_sht_amnt_cv > 0)
            GROUP BY
                banks.jst_code,
                banks.business_model,
                crdt.dt_rfrnc,
                inst.instrmnt_id,
                inst.dbtr_id,
                inst.cntrct_id;
            """

    return devo.read_sql(query)
