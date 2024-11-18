import pandas as pd
from typing import Union
from .ac_config import alignmentCalculatorConfig


def _harmonise_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    makes column names lowercase and connected by underscores instead of spaces

    Parameters
    ----------
    data : pd.DataFrame
        the DataFrame for which the columns names should be harmonised

    Returns
    -------
    pd.DataFrame
        the DataFrame with the columns names harmonised
    """

    data.columns = data.columns.str.replace(" ", "_")
    data.columns = data.columns.str.lower()

    return data


def _load_scenario_data(scenario_data: dict, allow_mismatches: bool = True) -> dict:
    """
    Load and process the scenario data

    Parameters
    ----------
    scenario_data : dict
        the scenario_data from the CLIMATE_SETTINGS as defined in parameters.py
    allow_mismatches : bool
        Flag indicating whether to allow scenarios to be formed based on different decarbonisation
        pathways if the sda data is missing for certain sectors
        default = True

    Returns
    -------
    dict
        A dict containing for each scenario set a dataframe with the regional classification
    """
    scenarios = dict()

    for year, data in scenario_data["scenario_files"].items():
        scenarios[year] = dict()
        for scenario_set, scenario in data.items():
            scen_tms_main = pd.read_csv(scenario["scenario_file_tms"])
            scen_sda_main = pd.read_csv(scenario["scenario_file_sda"])
            scenarios[year][scenario_set] = dict()

            if (
                allow_mismatches
                & (len(set(scen_tms_main["scenario"])) == 1)
                & (len(set(scen_sda_main["scenario"])) > 1)
            ):
                for scenario in set(scen_sda_main["scenario"]):
                    scen_tms = scen_tms_main.copy()
                    scen_tms["scenario"] = scenario
                    scen_sda = scen_sda_main.loc[
                        scen_sda_main["scenario"] == scenario, :
                    ]

                    df_scenario = pd.concat(
                        [
                            scen_tms.drop(columns=["scenario_source", "scenario"]),
                            scen_sda.drop(
                                columns=[
                                    "scenario_source",
                                    "scenario",
                                    "emission_factor_unit",
                                ]
                            ),
                        ]
                    )

                    df_scenario["technology"] = df_scenario["technology"].fillna("none")

                    scenarios[year][scenario_set][scenario] = _harmonise_column_names(
                        df_scenario
                    )

            elif (
                allow_mismatches
                & (len(set(scen_sda_main["scenario"])) == 1)
                & (len(set(scen_tms_main["scenario"])) > 1)
            ):
                for scenario in set(scen_tms_main["scenario"]):
                    scen_tms = scen_tms_main.loc[
                        scen_tms_main["scenario"] == scenario, :
                    ]
                    scen_sda = scen_sda_main.copy()
                    scen_sda["scenario"] = scenario

                    df_scenario = pd.concat(
                        [
                            scen_tms.drop(columns=["scenario_source", "scenario"]),
                            scen_sda.drop(
                                columns=[
                                    "scenario_source",
                                    "scenario",
                                    "emission_factor_unit",
                                ]
                            ),
                        ]
                    )

                    df_scenario["technology"] = df_scenario["technology"].fillna("none")

                    scenarios[year][scenario_set][scenario] = _harmonise_column_names(
                        df_scenario
                    )

            else:
                for scenario in set(scen_sda_main["scenario"]).intersection(
                    scen_tms_main["scenario"]
                ):
                    scen_tms = scen_tms_main.loc[
                        scen_tms_main["scenario"] == scenario, :
                    ]
                    scen_sda = scen_sda_main.loc[
                        scen_sda_main["scenario"] == scenario, :
                    ]

                    df_scenario = pd.concat(
                        [
                            scen_tms.drop(columns=["scenario_source", "scenario"]),
                            scen_sda.drop(
                                columns=[
                                    "scenario_source",
                                    "scenario",
                                    "emission_factor_unit",
                                ]
                            ),
                        ]
                    )

                    df_scenario["technology"] = df_scenario["technology"].fillna("none")

                    scenarios[year][scenario_set][scenario] = _harmonise_column_names(
                        df_scenario
                    )

    return scenarios


def _load_region_data(scenario_data: dict) -> dict:
    """
    Load and process the region data

    Parameters
    ----------
    scenario_data : dict
        the scenario_data from the CLIMATE_SETTINGS as defined in parameters.py

    Returns
    -------
    dict
        A dict containing for each scenario set a dataframe with the regional classification
    """

    regions = {}
    for scenario_set, region in scenario_data["region_file"].items():
        regions[scenario_set] = _harmonise_column_names(pd.read_csv(region))

    return regions


def _load_loanbook_data(loan_file: str|None) -> pd.DataFrame| None:
    """
    Load and process the loanbook data

    Parameters
    ----------
    loan_file : str |None
        the path to the loan file in csv format

    Returns
    -------
    pandas.DataFrame| None
        A dataframe containing the loan files as readin from the csv or
        None if the loan_file is None
    """
    if loan_file is None:
        return None
    else:
        loans = _harmonise_column_names(pd.read_csv(loan_file, index_col=0))

    return loans


def _preprocess_indicators(climate_data: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    excludes the inactive sectors from the climate_data

    Parameters:
    -----------
    climate_data: pd.DataFrame
        the climate company_indicators data as read in by the
        _load_main_climate_data function.

    settings: dict
        the settings equivalent to the settings from the parameters.py file
        that should apply when determining which sectors to exclude.
        default = None

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the indicators processed to exclude the inactive
        sectors.
    """
    sectors = []
    for sector, settings in settings["sectoral_approach"].items():
        if settings["active"]:
            sectors.append(sector)

    return climate_data[climate_data["sector"].isin(sectors)].copy()


def _load_main_climate_data(
    main_climate_file: Union[list, str],
    indicator_sheet: str = "Company Indicators - climate Comp",
    ownership_sheet: str = "Company Ownership",
    settings: dict|None = None,
) -> dict:
    """
    reads in the main climate data

    Parameters:
    -----------
    main_climate_file: list | str
        the name of the climate file or a list of climate files, in case
        the data is split into multiple files

    indicator_sheet: str, optional
        the name of the sheet company indicators that should be loaded
        default = 'Company Indicators - climate Comp'

    ownership_sheet: str, optional
        the name of the ownership sheet that should be loaded
        default = 'Company Ownership'

    settings: dict|None, optional
        the settings equivalent to the settings from the parameters.py file
        that should apply when loading the data. If None the settings
        from parameters.py will be used.
        default = None

    Returns
    -------
    dict
        A dict containing the company_idicators and the company_ownership
        dataframes.
    """
    if settings is None:
        settings = alignmentCalculatorConfig().load_settings()

    if isinstance(main_climate_file, str):
        with pd.ExcelFile(main_climate_file, engine="openpyxl") as xls:
            result = {
                "company_indicators": _harmonise_column_names(
                    _preprocess_indicators(
                        pd.read_excel(xls, indicator_sheet), settings
                    )
                ),
                "company_ownership": _harmonise_column_names(
                    pd.read_excel(xls, ownership_sheet)
                ),
            }

    elif isinstance(main_climate_file, list):
        if ".csv" in main_climate_file[0]:
            result = {
                "company_indicators": _harmonise_column_names(
                    _preprocess_indicators(pd.read_csv(main_climate_file[0]), settings)
                ),
                "company_ownership": _harmonise_column_names(
                    pd.read_csv(main_climate_file[1])
                ),
            }
        else:
            result = {
                "company_indicators": _harmonise_column_names(
                    _preprocess_indicators(
                        pd.read_excel(
                            main_climate_file[0], sheet_name="Company climate Dataset - EO"
                        ),
                        settings,
                    )
                ),
                "company_ownership": _harmonise_column_names(
                    pd.read_excel(main_climate_file[1], sheet_name="Company Ownership")
                ),
            }
    else:
        raise ValueError()

    return result
