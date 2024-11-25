import pandas as pd
import numpy as np
from typing import Any, List
from .load_climate_data import (
    _load_main_climate_data,
    _load_scenario_data,
    _load_loanbook_data,
    _load_region_data,
)
from .prepare_loanbook import loanbookPreparer
from .ac_config import alignmentCalculatorConfig
from .alignment_results import alignmentResults


class alignmentCalculator:
    """
    The alignmentCalculator can calculate the alignment of a loan book. The
    alignmentCalculator uses the default parameters for the parameters.py file.

    Parameters
    ----------
    portfolio_id : str, optional
        Identifier for the loan portfolio (default is "portfolio_code").
    custom_settings : dict| None, optional
        Custom settings dictionary to update default settings (default is None).
        For an overview of which settings could be given, please see the
        docstring of the update_settings method.
    loan_file : str or pandas.DataFrame or bool, optional
        The loan file to be used. Depending on the type of the loan_file it is assumed
        to have a certain value:
            - str: the path to a loan file.
            - pd.dataFrame: a dataframe containing the loan data.
            - True: the loan data will be generated from a loanbookPreparer
            - False: no loan data will be loaded.
        default is False
    loanbook_settings : dict, optional
        Settings for loanbook creation, for a full overview of the settings
        please see the docsting of the loanbookPreparer here also you will find the
        default values the loanbookPreparer uses. (default is {
            'base_year': 2023,
            'month': 12,
            'start_year': None,
            'start_month': None,
            'frequency': 'Y',
            'additional_columns': None,
            'external_columns': ['total_assets', 'total_debt', 'turnover']
        }).
    scenario_set : str, optional
        Scenario set for loan processing (default is "weo").
    pathway : str, optional
        Pathway identifier (default is "nze_2050").
    debug : bool, optinal
        Flag to contstruct the alignment calculator in debug mode. In debug
        mode, the initialisation is not performed.

    Methods
    -------
    calculate_net_alignment(loan_indicator: str = "outstanding_amount",
                            facet_col: List[str] = [],
                            bopo_split: bool = False,
                            individual_loans: bool = False,
                            use_loan_file: bool = True,
                            use_region_file: bool = True,
                            only_parents: bool = True,
                            limit: int = 3)
        Calculates the net alignment.
    calculate_net_alignment_change_over_time(loan_indicator: str = "outstanding_amount",
                                                facet_col: List[str] = [],
                                                bopo_split: bool = False,
                                                only_parents: bool = True,
                                                use_region_file: bool = True,
                                                limit: int = 3,
                                                add_total: bool = True)
        Determines the change in alignment from one time period to the next.
    get_available_scenarios()
        Get a list of the available scenarios that have been loaded in by the
        alignmentCalculator.
    set_scenarios(scenario_set: str, pathway: str):
        Set the scenario to a specific value. This scenario will be used next
        time the calculate_net_alignment or calculate_net_alignment_change_over_time
        method will run.
    update_loanbook(base_year: int = 2022,
                    month: int = 12,
                    loan_file: str = "",
                    additional_instrument_columns: dict = None)
        Generates a new loanbook in memory that can be used for subsequent net
        alignment calculations.
    update_settings(settings_change: dict):
        Updates the settings file for the alignmentCalculator after it has been
        constructed.
    """

    def __init__(
        self,
        portfolio_id: str = "portfolio_code",
        custom_settings: dict | None = None,
        loan_file: str| pd.DataFrame| bool = False,
        loanbook_settings: dict = {
            "base_year": 2023,
            "month": 12,
            "start_year": None,
            "start_month": None,
            "frequency": "Y",
            "additional_columns": None,
            "external_columns": ["total_assets", "total_debt", "turnover"],
        },
        scenario_set: str = "weo",
        pathway: str = "nze_2050",
        debug: bool = False,
    ) -> None:

        self._portfolio_id = portfolio_id
        self._settings = alignmentCalculatorConfig().load_settings()
        if custom_settings is not None:
            self.update_settings(custom_settings)

        self._climate_company_indicators = {}
        self._climate_ownership = {}
        self._df_climate = {}
        self._external_columns = loanbook_settings["external_columns"]
        self._scenario_set = scenario_set
        self._pathway = pathway
        self._regions = None

        if not debug:
            self._scenario_data = _load_scenario_data(self._settings["scenario_data"])

            for year in self._settings["main_climate_file"].keys():
                climate_data = _load_main_climate_data(
                    self._settings["main_climate_file"][year], settings=self._settings
                )
                self._climate_company_indicators[year] = climate_data["company_indicators"]
                self._climate_ownership[year] = climate_data["company_ownership"]

            if isinstance(loan_file, str):
                self._loans = _load_loanbook_data(loan_file)
            elif isinstance(loan_file, pd.DataFrame):
                self._loans = loan_file.copy()
            elif loan_file == True:
                lbm = loanbookPreparer(settings=self._settings)
                self._loans = lbm.prepare_loanbook(**loanbook_settings)

    def calculate_net_alignment(
        self,
        loan_indicator: str = "outstanding_amount",
        facet_col: List[str] = [],
        bopo_split: bool = False,
        individual_loans: bool = False,
        use_loan_file: bool = True,
        only_parents: bool = True,
        use_region_file: bool = True,
        limit: int = 3,
        normalise_method: str = "total",
    ) -> alignmentResults:
        """
        Calculate aggregated net alignment results based on input parameters.

        Parameters
        ----------
        loan_indicator : str, optional
            The loan indicator to use for calculations.
            Default is 'outstanding_amount'.

        facet_col : List[str], optional
            List of columns to use for facet-based aggregation.
            Default is an empty list.

        bopo_split : bool, optional
            Flag to enable build-out phase-out split calculations.
            Default is False.

        individual_loans : bool, optional
            Flag indicating whether to process individual loans.
            Default is False.

        use_loan_file : bool, optional
            Flag indicating whether to use loan file data.
            Default is True.

        only_parents : bool, optional
            Flag indicating whether to calculate on the parent level rather than company level.
            Default is True.

        use_region_file : bool, optional
            Flag indicating whether to use region file data.
            Default is True.

        limit : int, optional
            Limit for score values.
            Default is 3.

        normalise_method : str, optional
            Method of normalisation to apply:
            - "global": Perform global normalisation based on sector-wise production totals.
            - "company": Perform company-level normalisation based on sector-wise production totals.
            - "portfolio": Perform normalisation based on portfolio data.
            - "total": Perform normalisation based on the combined data (default).
            - "economic": Perform normalisation using economic weights.
            default is "total"

        Returns
        -------
        alignmentResults
            The results of the net alignment calculations as an alignmentResults object.
        """

        if isinstance(facet_col, str):
            facet_col = [facet_col]

        master_data = {}
        portfolio_dates = set()
        for year in self._climate_company_indicators.keys():
            if (self._scenario_set in self._scenario_data[year]) and (
                self._pathway in self._scenario_data[year][self._scenario_set]
            ):
                master_data[year] = self._preprocess_data(
                    use_loan_file,
                    individual_loans,
                    loan_indicator,
                    only_parents,
                    facet_col,
                    use_region_file,
                    year,
                    normalise_method,
                )
                portfolio_dates.update(
                    set(master_data[year]["portfolio_date"].unique().tolist())
                )

        total_results = []
        ar = alignmentResults(
            pd.DataFrame(),
            self._climate_company_indicators,
            self._df_climate,
            self._scenario_data,
            self._settings,
            self._portfolio_id,
        )
        for portfolio_date in portfolio_dates:
            if int(portfolio_date / 100) in master_data.keys():
                data = master_data[int(portfolio_date / 100)].copy()
            else:
                print(
                    f"{portfolio_date} was not matched to climate data in {int(portfolio_date/100)}"
                )
                continue

            results = data.loc[data["portfolio_date"] == portfolio_date, :].copy()

            total_results.append(
                self._calculate_alignment_instance(
                    results, facet_col, loan_indicator, bopo_split, limit
                )
            )

            ar = alignmentResults(
                pd.concat(total_results),
                self._climate_company_indicators,
                self._df_climate,
                self._scenario_data,
                self._settings,
                self._portfolio_id,
            )

        return ar

    def calculate_net_alignment_change_over_time(
        self,
        loan_indicator: str = "outstanding_amount",
        facet_col: List[str] = [],
        bopo_split: bool = False,
        only_parents: bool = True,
        limit: int = 3,
        normalise_method="total",
        add_total: bool = True,
        use_region_file: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate net alignment change over time.

        Parameters:
        -----------
        loan_indicator : str
            column with the loan amount relevant for the analysis.
            Default is "outstanding_amount".

        facet_col : List[str], optional
            List of columns to use as facets in the analysis.
            Default is an empty list.

        bopo_split : bool, optional
            Whether to split based on build-out/phase-out.
            Default is False.

        only_parents : bool, optional
            Whether to consider only parent companies.
            Default is True.

        limit : int, optional
            The limit for alignment change calculation.
            Default is 3.

        normalise_method : str, optional
            Method of normalisation to apply:
            - "global": Perform global normalisation based on sector-wise production totals.
            - "company": Perform company-level normalisation based on sector-wise production totals.
            - "portfolio": Perform normalisation based on portfolio data.
            - "total": Perform normalisation based on the combined portfolio data (default).
            - "economic": Perform normalisation using economic weights.
            default is "total"

        add_total : bool, optional
            Whether to add a total to the results.
            Default is True.

        use_region_file : bool, optional
            Whether to use region file in preprocessing.
            Default is True.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the calculated net alignment change over time.
        """
        if isinstance(facet_col, str):
            facet_col = [facet_col]

        facet_col = facet_col + ["data_year", "scenario_year"]

        portfolio_dates = self._make_portfolio_dates()
        master_data = self._make_master_data(
            loan_indicator, only_parents, facet_col, use_region_file, normalise_method
        )

        total_results = []
        for portfolio_date in portfolio_dates:
            for scenario_year, data_scenario_year in master_data.items():
                if -1 < (int(portfolio_date / 100) - scenario_year) < 2:
                    for year, data_year in data_scenario_year.items():
                        data = data_year.loc[
                            data_year["portfolio_date"] == portfolio_date, :
                        ].copy()
                        if -2 < (int(portfolio_date / 100) - year) < 1:
                            if (int(portfolio_date / 100) - year) == -1:
                                data["year"] = data["year"] - 1
                            if ((int(portfolio_date / 100) - year) == 0) & (
                                (int(portfolio_date / 100) - scenario_year) == 1
                            ):
                                data["year"] = data["year"] - 1
                                data["portfolio_date"] = data["portfolio_date"] - 50

                            data["data_year"] = year
                            data["scenario_year"] = scenario_year

                            total_results.append(
                                self._calculate_alignment_instance(
                                    data, facet_col, loan_indicator, bopo_split, limit
                                )
                            )

        total_results = pd.concat(total_results)
        total_results.loc[
            (total_results["portfolio_date"] % 100 >= 50), "portfolio_date"
        ] = (
            total_results.loc[
                (total_results["portfolio_date"] % 100 >= 50), "portfolio_date"
            ]
            + 50
        )

        return self._aggregate_over_time_results(
            total_results, loan_indicator, add_total
        )

    def get_available_scenarios(self) -> pd.DataFrame:
        """
        Get available scenarios from the loaded scenario data.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing available scenarios with columns:
            - 'scenario_set': Scenario set name.
            - 'pathway': Pathway identifier.
            - 'year': Year of the scenario.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> scenarios = ac.get_available_scenarios()
        >>> print(scenarios)
            scenario_set    pathway  year
        0            weo  nze_2050  2022
        1            weo  nze_2050  2023
        ...          ...       ...   ...
        """
        results = dict()
        results["scenario_set"] = []
        results["pathway"] = []
        results["year"] = []

        for year, data in self._scenario_data.items():
            for scenario_set, scenario_set_data in data.items():
                for pathway in scenario_set_data.keys():
                    results["scenario_set"].append(scenario_set)
                    results["pathway"].append(pathway)
                    results["year"].append(year)

        return pd.DataFrame(results)

    def set_scenarios(self, scenario_set: str, pathway: str) -> None:
        """
        Set the scenario set and pathway for loan processing.

        Parameters
        ----------
        scenario_set : str
            Scenario set name to use for loan processing.
        pathway : str
            Pathway identifier to use for loan processing.

        Returns
        -------
        None

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> ac.set_scenarios("weo", "nze_2050")
        """
        self._scenario_set = scenario_set
        self._pathway = pathway

    def update_settings(self, settings_change: dict, save_changes: bool = True) -> None:
        """
        Update settings of the alignmentCalculator instance with new values.

        Parameters
        ----------
        settings_change : dict
            Dictionary containing the settings to update. Each key-value pair
            corresponds to a setting parameter and its new value. The settings
            is a dict possibly containing the following entries:

            main climate file
            The link to the climate files as supplied by AI. The year key is the start_year of the scenario.
            company_analytics_file_location should be a string with the file_location of the company_analytics
            file and company_indicators_file_location the file location of the company_inidicators file.
            "main_climate_files":
                {year: [company_analytics_file_location,
                        company_indicators_file_location],
                },

            company information file
            The location of the company information from AI, company_information_file_location should
            be a string pointing to the location of the company information file.
            "company_information_file": company_information_file_location,

            economic weights
            The economic weights are used for the sector normalisation of the alignment scores. The economic
            weights should be given for all sectors. In the economic weights dict the sector key indicates
            which of the sectors the weights should be applied to. The weight should be an numeric value that
            indicates the weighing of the sector.
            "economic_weights":
                {sector: weight,
                },

            production thresholds
            The production thresholds are used to determine which companies are filtered out of the data
            because the production-turnover ratio or the production-total asset ratio is too low. The sector
            key indicates which sector the ratios apply to, the ar_ratio is the production-total asset ratio
            abd the to_ratio is the production-turnover ratio.
            "production_thresholds": {
                sector: {
                    "asset_ratio": ar_ratio, "turnover_ratio": to_ratio
                    },
            },

            scenario data
            The scenario data includes links to the scenario files and the regions used for the scenario data.
            The scenario files should be given per year, this year should also be the start year of the
            scenario and will be the start year of the analysis. The scenario_set_name is the name of the
            set of scenarios described in the files and the scenario_file_tms and scenario_file_sda are the
            file locations of the tms and sda scenarios for the scenario set. The region_file_location refers
            to the region file for the scenario set.
            "scenario_data": {
                "scenario_files": {
                    start_year: {
                        scenario_set_name: {
                            scenario_file_tms,
                            scenario_file_sda,
                        },
                },
                "region_file": {
                    scenario_set_name: region_file_location
                },
            },

            sectoral approach
            The sectoral approach determines how each of the sectors should be approached during the
            calculation of the net alignment. The sector key indicates for which sectors the following
            values will hold. The tms_sda value should be either "tms" or "sda" dependent on which
            approach should be taken for the calculation. For each of the technologies within the
            sector, e.g. technology_a, technology_b, technology_c and technology_d it should be
            indicated whether the sector based targets or technology based targets should be used if
            the sector is a tms sector. The build_out, phase_out and other keys indicate whether the
            technology should be seen as a build_out, phase_out or other technology. For the sda
            technologies all the technologies should be other technologies. The regional_bool
            indicates whether regional data should be used for the calculation of the targets
            and the active bool indicates whether or not a sector is active and will be taken into
            account when reading the data.
            "sectoral_approach": {
                sector: {
                    "approach": tms_sda,
                    "sector": [technology_b],
                    "technology": [technology_a, technology_b, technology_c, technology_d],
                    "build_out": [technology_b],
                    "phase_out": [technology_a, technology_c],
                    "other": [technology_d],
                    "regional": regional_bool,
                    "active": active_bool,
                },

        settings_change : bool
            Flag whether to save the adjusted settings in the users settings file.
            default = True


        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> ac.update_settings({
                "company_information_file":
                    'C:\\data\\climate_data\\company_information.csv'
                })
        """
        main_climate_file = None
        company_information_file = None
        economic_weights = None
        production_thresholds = None
        scenario_data = None
        sectoral_approach = None
        for key, val in settings_change.items():
            if key == "main_climate_file":
                main_climate_file = val
            elif key == "company_information_file":
                company_information_file = val
            elif key == "economic_weights":
                economic_weights = val
            elif key == "production_thresholds":
                production_thresholds = val
            elif key == "scenario_data":
                scenario_data = val
            elif key == "sectoral_approach":
                sectoral_approach = val
            else:
                raise ValueError(
                    f"the given key: {key} of the settings_change dict "
                    + "is not a valid option, please see the "
                    + "docstring for all valid options"
                )

        new_settings = alignmentCalculatorConfig().config(
            main_climate_file=main_climate_file,
            company_information_file=company_information_file,
            economic_weights=economic_weights,
            production_thresholds=production_thresholds,
            scenario_data=scenario_data,
            sectoral_approach=sectoral_approach,
            save_changes=save_changes,
        )

        if new_settings is None:
            self._settings = alignmentCalculatorConfig().load_settings()
        else:
            self._settings = new_settings

    def update_loanbook(
        self, loan_file: str | None = None, loanbook_settings: dict | None = None
    ) -> None:
        """
        Update the loanbook data based on the specified parameters.

        Parameters
        ----------
        loan_file : str |None, optional
            Path to a loan file containing loan data (default is None).
            If provided, loads loan data from the specified file.
        loanbook_settings : dict |None, optional
            The loanbook_settings dict can contain the following key, value
            pairs.
            base_year : int
                The base year of the loan data.
            month : int
                The month for which loan data will be loaded.
            start_year : int
                The year from which to start loading the loan data.
            start_month : int
                The month from which to start loading the loan data.
            portfolio_codes : list
                List of portfolio codes to filter the loan data.
            loanbook_filename : str
                If provided, the loanbook data will be saved as a CSV file with the
                specified filename. If not provided, the loanbook DataFrame will be
                returned without saving.
            frequency: str
                String indicating what the frequency of the fetched data should be,
                - 'M': Monthly
                - 'Q': Quarterly
                - 'Y': Yearly
            additional_columns: dict
                Additional columns that should be loaded from the loan data database.
            external_columns: list
                List indicating which external columns should be added, in case none is
                provided no connection to an external data provider will be made.

        Returns
        -------
        None

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> ac.update_loanbook(base_year=2023, month=1)
        >>> ac.update_loanbook(loan_file="loans.csv")
        """
        if loan_file is None:
            lbm = loanbookPreparer()
            if loanbook_settings is None:
                loanbook_settings = {}
            self._loans = lbm.prepare_loanbook(**loanbook_settings)
        else:
            self._loans = _load_loanbook_data(loan_file)

    def _preprocess_data(
        self,
        use_loan_file: bool,
        individual_loans: bool,
        loan_indicator: str,
        only_parents: bool,
        facet_col: list,
        use_region_file: bool,
        year: int,
        normalise_method: str,
        scenario_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Preprocesses data for loan processing based on configuration.

        This method preprocesses data for loan processing based on the specified
        configuration options and prepares the DataFrame for further analysis.

        Parameters
        ----------
        use_loan_file : bool
            Flag indicating whether to use loan file data.
        individual_loans : bool
            Flag indicating whether to consider individual loans, or whether loans
            can be grouped at unique company-protfolio_id relation.
        loan_indicator : str
            column with the loan amount relevant for the analysis.
        only_parents : bool
            Flag indicating whether to consider only parent companies.
        facet_col : List[str]
            List of facet columns or single column used in the net alignment
            calculation.
        use_region_file : bool
            Flag indicating whether to specify targets on a regional basis if
            this is indicated in the settings under the sectoral_approach.
        year : int
            The year for which the produciton data is being processed.
        normalise_method : str
            Method of normalisation to apply:
            - "global": Perform global normalisation based on sector-wise production totals.
            - "company": Perform company-level normalisation based on sector-wise production totals.
            - "portfolio": Perform normalisation based on the individual portfolio data.
            - "total": Perform normalisation based on total loan data given.
            - "economic": Perform normalisation using economic weights.
        scenario_year : int|None, optional
            The scenario year used for calculations, if none is given it will
            be set equal to year, by default None.

        Returns
        -------
        pandas.DataFrame
            Preprocessed DataFrame ready for loan processing.

        Examples
        --------
        >>> lp = LoanProcessor()
        >>> preprocessed_data = lp._preprocess_data(
        ...     use_loan_file=True,
        ...     individual_loans=False,
        ...     loan_indicator='outstanding_amount',
        ...     only_parents=True,
        ...     facet_col=['sector'],
        ...     use_region_file=True,
        ...     year=2023,
        ...     normalise_method='portfolio',
        ... )
        """
        if use_region_file:
            self._regions = _load_region_data(self._settings["scenario_data"])

        if scenario_year is None:
            scenario_year = year

        self._calculate_climate(year, scenario_year)
        climate_data = self._combine_asset_locations(year)
        climate_data = self._combine_climate_loan_data(
            climate_data,
            use_loan_file,
            individual_loans,
            only_parents,
            loan_indicator,
            facet_col,
            year,
        )

        climate_data = self._apply_production_thresholds(climate_data)
        climate_data = self._split_loans_over_sector(climate_data, loan_indicator, year)
        climate_data = self._split_over_technology(climate_data, loan_indicator)

        climate_data = self._normalise_production(
            normalise_method,
            climate_data,
            year,
            self._settings["economic_weights"],
            loan_indicator=loan_indicator,
        )
        climate_data["target"] = climate_data["target"] / climate_data["norm"]
        climate_data["production"] = climate_data["production"] / climate_data["norm"]

        return climate_data

    def _calculate_climate(self, year: int, scenario_year: int) -> None:
        """
        Calculate climate targets and update company indicators for a specific year.

        This method calculates climate targets and updates company indicators for
        a specific year based on the provided scenario year.

        Parameters
        ----------
        year : int
            The year for which climate targets are being calculated.
        scenario_year : int
            The scenario year used for calculations.

        Returns
        -------
        None

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> ac._calculate_climate(year=2023, scenario_year=2024)
        """

        self._add_region(year, self._reconcile_regions())
        df_climate = self._climate_company_indicators[year].copy()
        df_climate['production'] = df_climate['production'].astype(np.float64)
        df_climate["technology_temp"] = df_climate["technology"]

        for sector, sector_settings in self._settings["sectoral_approach"].items():
            if sector_settings["approach"] == "sda":
                df_climate.loc[(df_climate["sector"] == sector), "technology_temp"] = "none"

        df_climate = df_climate.merge(
            self._scenario_data[scenario_year][self._scenario_set][self._pathway],
            how="left",
            left_on=["sector", "technology_temp", "year", "region"],
            right_on=["sector", "technology", "year", "region"],
            suffixes=["", "_scenario"],
        )
        df_climate["target"] = np.nan
        df_climate["target"] = df_climate["target"].astype(np.float64)

        for sector, sector_settings in self._settings["sectoral_approach"].items():
            if sector_settings["approach"] == "tms":
                df_climate = self._calculate_tms(
                    df_climate, sector, sector_settings, scenario_year
                )
            elif sector_settings["approach"] == "sda":
                df_climate = self._calculate_sda(df_climate, sector, scenario_year)

        df_climate = df_climate[
            [
                "company_id",
                "name_company",
                "sector",
                "technology",
                "plant_location",
                "year",
                "production",
                "emission_factor",
                "region",
                "target",
            ]
        ]

        df_climate.loc[df_climate["emission_factor"] > 0, "target"] = (
            df_climate.loc[df_climate["emission_factor"] > 0, "target"]
            * df_climate.loc[df_climate["emission_factor"] > 0, "production"]
        )
        df_climate.loc[df_climate["emission_factor"] > 0, "production"] = (
            df_climate.loc[df_climate["emission_factor"] > 0, "emission_factor"]
            * df_climate.loc[df_climate["emission_factor"] > 0, "production"]
        )

        self._df_climate[year] = df_climate.dropna(subset=["target"])

    def _add_region(self, year: int, region_mapping: dict | None = None) -> None:
        """
        Add region information to company indicators for a specific year.

        This method adds region information to company indicators for a given year
        based on the provided region mapping. If None is given as a region_mapping
        no action is performed.

        Parameters
        ----------
        year : int
            The year for which region information is being added.
        region_mapping : dict|None, optional
            Dictionary mapping sectors to region-country mappings, by default None.

        Examples
        --------
        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> ac._calculate_climate(year=2023, scenario_year=2024)
        >>> ac._add_region(2023, ac._reconsile_regions())
        """
        self._climate_company_indicators[year]["region"] = "global"

        if region_mapping is not None:
            for sector_name, sector_region in region_mapping.items():
                sector_rows = (
                    self._climate_company_indicators[year]["sector"] == sector_name
                )
                for region, countries in sector_region.items():
                    region_rows = self._climate_company_indicators[year][
                        "plant_location"
                    ].isin(countries)
                    self._climate_company_indicators[year].loc[
                        sector_rows & region_rows, "region"
                    ] = region

    def _reconcile_regions(self) -> dict | None:
        """
        Reconcile regional data based on sectoral approaches.

        This method reconciles regional data based on sectoral approaches defined
        in the settings.

        Returns
        -------
        dict|None
            Dictionary mapping sectors to region-country mappings, if the attribute
            _regions has not yet been initiated, None is returned


        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> ac._calculate_climate(year=2023, scenario_year=2024)
        >>> ac._reconcile_regions()
        'coal': {'global': ['BG','HR','CY','MT', ...
        """

        if self._regions is None:
            return None

        regions = self._regions[self._scenario_set]

        options = self._scenario_data[list(self._scenario_data.keys())[0]][
            self._scenario_set
        ][self._pathway].merge(
            regions, how="inner", left_on="region", right_on="region"
        )
        options = (
            options.groupby(["sector", "region"], as_index=False)
            .count()
            .sort_values(by=["isos"])
            .reset_index(drop=True)
        )

        region_mapping = {}

        for sector, region in options[["sector", "region"]].values:
            if self._settings["sectoral_approach"][sector]["regional"]:
                country_list = (
                    regions.loc[regions["region"] == region, "isos"]
                    .str.upper()
                    .to_list()
                )

                if sector in region_mapping.keys():
                    assigned_countries = []
                    for countries in region_mapping[sector].values():
                        assigned_countries = assigned_countries + countries
                    country_list = list(set(country_list) - set(assigned_countries))
                    region_mapping[sector][region] = country_list

                else:
                    region_mapping[sector] = {region: country_list}
            else:
                country_list = (
                    regions.loc[regions["region"] == "global", "isos"]
                    .str.upper()
                    .to_list()
                )
                region_mapping[sector] = {region: country_list}

        return region_mapping

    def _calculate_tms(
        self,
        df_climate: pd.DataFrame,
        sector: str,
        sector_settings: dict,
        scenario_year: int,
    ) -> pd.DataFrame:
        """
        Calculate and update target values based on TMS calculation for a sector.

        Parameters
        ----------
        df_climate : pandas.DataFrame
            DataFrame containing company indicators data (climate data).
        sector : str
            Sector name for which TMS calculation is performed.
        sector_settings : dict
            Sector settings dictionary containing sector-specific settings.
        scenario_year : int
            Scenario year for which the calculation is performed.

        Returns
        -------
        pandas.DataFrame
            Updated DataFrame with calculated target values.
        """

        for tech in sector_settings["sector"]:
            df_totals = df_climate.merge(
                df_climate[df_climate["year"] == scenario_year]
                .groupby(["company_id", "sector", "region"], as_index=False)[
                    "production"
                ]
                .sum(),
                how="left",
                left_on=["company_id", "sector", "region"],
                right_on=["company_id", "sector", "region"],
                suffixes=("", "_ini_total"),
            )
            df_totals = df_totals.merge(
                df_climate[
                    (df_climate["year"] == scenario_year)
                    & (df_climate["technology"] == tech)
                ]
                .groupby(["company_id", "sector", "region"], as_index=False)[
                    "production"
                ]
                .sum(),
                how="left",
                left_on=["company_id", "sector", "region"],
                right_on=["company_id", "sector", "region"],
                suffixes=("", "_ini"),
            )
            rows = (df_climate["technology"] == tech) & (df_climate["sector"] == sector)
            df_climate.loc[rows, "target"] = (
                df_totals.loc[rows, "production_ini_total"] * df_climate.loc[rows, "smsp"]  # type: ignore
                + df_totals.loc[rows, "production_ini"]
            )

        for tech in sector_settings["technology"]:
            df_initial = df_climate.merge(
                df_climate[df_climate["year"] == scenario_year]
                .groupby(
                    ["company_id", "plant_location", "technology", "sector", "region"],
                    as_index=False,
                )["production"]
                .sum(),
                how="left",
                left_on=[
                    "company_id",
                    "plant_location",
                    "sector",
                    "technology",
                    "region",
                ],
                right_on=[
                    "company_id",
                    "plant_location",
                    "sector",
                    "technology",
                    "region",
                ],
                suffixes=("", "_ini"),
            )
            rows = (df_climate["technology"] == tech) & (df_climate["sector"] == sector)
            df_climate.loc[rows, "target"] = (
                df_initial.loc[rows, "production_ini"] * df_climate.loc[rows, "tmsr"]  # type: ignore
            )

        return df_climate

    def _calculate_sda(
        self, df_climate: pd.DataFrame, sector: str, scenario_year: int
    ) -> pd.DataFrame:
        """
        Calculate and update target values based on SDA calculation for a sector.

        Parameters
        ----------
        df_climate : pandas.DataFrame
            DataFrame containing company indicators data (climate data).
        sector : str
            Sector name for which SDA calculation is performed.
        scenario_year : int
            Scenario year for which the calculation is performed.

        Returns
        -------
        pandas.DataFrame
            Updated DataFrame with calculated target values.
        """

        df_climate.loc[df_climate["sector"] == sector, "target"] = df_climate.loc[
            df_climate["sector"] == sector, "emission_factor_scenario"
        ]

        return df_climate

    def _combine_asset_locations(self, year: int) -> pd.DataFrame:
        """
        Combine and aggregate data from the climate DataFrame based on the location of the assets.
        Parameters
        ----------
        year : int
            The year for which data should be processed.

        Returns
        -------
        pd.DataFrame
            Combined and aggregated DataFrame based on the location of the assets.
        """

        data = self._df_climate[year].copy()
        sda_rows = data["emission_factor"] > 0
        data.loc[sda_rows, "technology"] = data.loc[sda_rows, "sector"]

        group_columns = [
            "company_id",
            "name_company",
            "sector",
            "technology",
            "year",
            "region",
        ]
        agg_columns = ["production", "target"]

        sector_approaches = self._get_sector_approach_technologies()
        sector_approach_rows = data["technology"].isin(sector_approaches)
        data_technology = (
            data[~sector_approach_rows]
            .groupby(group_columns, as_index=False)[agg_columns]
            .sum()
        )
        data_sector = (
            data[sector_approach_rows]
            .groupby(group_columns, as_index=False)[agg_columns]
            .agg({"production": "sum", "target": "mean"})
        )

        return pd.concat([data_technology, data_sector])

    def _get_sector_approach_technologies(self) -> list:
        """
        Get the list of technologies associated with sectoral approaches defined in settings.

        Returns
        -------
        list
            List of technologies associated with sectoral approaches.
        """
        sector_approaches = []

        for _, sector_settings in self._settings["sectoral_approach"].items():
            for sector_approach in sector_settings["sector"]:
                sector_approaches.append(sector_approach)

        return sector_approaches

    def _combine_climate_loan_data(
        self,
        climate_data: pd.DataFrame,
        use_loan_file: bool,
        individual_loans: bool,
        only_parents: bool,
        loan_column: str,
        facet_col: list,
        year: int,
    ) -> pd.DataFrame:
        """
        Combine climate TMS and SDA data with loan data and normalise the combined data.

        This method combines climate TMS (Total Market Share) and SDA (Sectoral Decarbonisation)
        data with loan data and subsequently normalises the combined data.

        Parameters
        ----------
        climate_data : pandas.DataFrame
            DataFrame containing climate data (TMS and SDA).
        use_loan_file : bool
            Flag indicating whether to use loan file data.
        individual_loans : bool
            Flag indicating whether to consider individual loans.
        only_parents : bool
            Flag indicating whether to consider only parent companies.
        loan_column : str
            column with the loan amount relevant for the analysis.
        facet_col : list of str
            List of facet columns or single column used for grouping
            and aggregation.
        year : int
            The year for which data is being combined.

        Returns
        -------
        pandas.DataFrame
            Combined and normalised DataFrame.
        """
        if use_loan_file:
            if self._loans is not None:
                loan_data = self._loans.copy()
            else:
                raise ValueError("no loandata has been supplied")
            if only_parents:
                loan_data = self._only_parents(
                    loan_data, loan_column, year, facet_col=facet_col
                )
            if not individual_loans:
                grouper = set(loan_data.columns) & set(
                    [
                        self._portfolio_id,
                        "portfolio_date",
                        "dbtr_id",
                        "company_name",
                        "name_company",
                        "business_model",
                        "company_country",
                        "company_lei",
                        "parent_name",
                        "parent_lei",
                        "bvdidnumber",
                        "company_id",
                    ]
                    + facet_col
                    + self._external_columns
                )
                loan_data = loan_data.groupby(
                    list(grouper), as_index=False, dropna=False
                ).agg(
                    {
                        loan_column: "sum",
                        "loan_id": lambda x: str(hash("".join(x.values))),
                    }
                )
        else:
            loan_data = self._df_climate[year][["company_id"]].copy()
            loan_data[self._portfolio_id] = "all"
            loan_data["loan_id"] = loan_data["company_id"]
            loan_data[loan_column] = 1
            if only_parents:
                loan_data = self._only_parents(loan_data, loan_column, year)
            loan_data[loan_column] = 1

        for extra_col in facet_col:
            if extra_col != "company_id":
                if (extra_col in climate_data.columns) and (
                    extra_col in loan_data.columns
                ):
                    climate_data = climate_data.rename(
                        columns={extra_col: extra_col + "_climate"}
                    )
        climate_data = loan_data.merge(
            climate_data, how="left", left_on=["company_id"], right_on=["company_id"]
        )
        if "portfolio_date" not in climate_data.columns:
            climate_date = []
            for date in self._climate_company_indicators.keys():
                climate_data["portfolio_date"] = int(str(date) + "12")
                climate_date.append(climate_data.copy())
            climate_data = pd.concat(climate_date)

        return climate_data

    def _only_parents(
        self,
        loans: pd.DataFrame,
        loan_indicator: str,
        year: int,
        facet_col: list = [],
        stop_at_weak_parents: bool = True,
    ) -> pd.DataFrame:
        """
        Convert loanbook data to include only parent companies.

        Parameters:
        -----------
        loans : pd.DataFrame
            DataFrame representing the loanbook with possible different portfolios.

        loan_indicator : str
            The loan indicator used for aggregation (e.g., 'outstanding_amount').

        year : int
            The year to consider for retrieving parent companies.

        facet_col : list, optional
            List of columns used for grouping loans.
            Default is an empty list.

        stop_at_weak_parents : bool, optional
            If True, only include parent companies that are not weak parents.
            If False, consider all parent companies that are ultimate listed or ultimate parents.
            Default is True.

        Returns:
        --------
        pd.DataFrame
            Adjusted loanbook DataFrame with loans aggregated by parent company.
        """

        parents = self._get_parent_companies(stop_at_weak_parents, year)
        grouper_cols = list(
            set(
                ["company_id", "loan_id", self._portfolio_id, "portfolio_date"]
                + facet_col
            )
            & set(loans.columns)
        )
        # Set the company ids to the parent company ids
        loans = loans[loans["company_id"].isin(parents.index)].copy()
        loans["company_id"] = parents.loc[
            loans["company_id"], "parent_company_id"
        ].values
        agg_function = {
            **{loan_indicator: "sum"},
            **{
                col: "first"
                for col in loans.columns.drop([loan_indicator] + grouper_cols)
            },
        }
        loans = loans.groupby(grouper_cols, as_index=False).agg(agg_function)

        return loans

    def _get_parent_companies(
        self, stop_at_weak_parents: bool, year: int, lowest_level: bool = True
    ) -> pd.DataFrame:
        """
        Get parent companies from the climate data.

        Parameters:
        -----------
        stop_at_weak_parents : bool
            If True, restrict to parent companies that are not weak parents.
            If False, consider all parent companies that are ultimate listed or ultimate parents.

        year : int
            The year for which parent companies are retrieved.

        lowest_level : bool, optional
            If True, get parent companies at the lowest ownership level.
            If False, get parent companies at the highest ownership level.
            Default is True.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing parent companies with their corresponding parent_company_id s.
        """
        df_structure = self._climate_ownership[year].copy()

        if stop_at_weak_parents:
            df_structure = df_structure[df_structure["is_parent"]]
        else:
            df_structure = df_structure[
                (df_structure["is_parent"])
                & (
                    df_structure["is_ultimate_listed_parent"]
                    | df_structure["is_ultimate_parent"]
                )
            ]
        df_structure = df_structure.sort_values(
            by="ownership_level", ascending=lowest_level
        )
        parents = pd.DataFrame(
            df_structure.groupby("company_id")["parent_company_id"].first()
        )

        df_structure = self._climate_ownership[year].copy()
        others = df_structure[
            ~df_structure["company_id"].isin(parents.index)
        ].sort_values(by="ownership_level", ascending=False)
        others = pd.DataFrame(others.groupby("company_id")["parent_company_id"].first())

        return pd.concat([parents, others])

    def _apply_production_thresholds(self, loan_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply production thresholds to filter eligible companies in the loan portfolio.

        This method applies production thresholds to filter eligible companies in the
        loan portfolio data based on turnover-production and total asset-production ratios.

        Parameters
        ----------
        loan_data : pandas.DataFrame
            DataFrame containing loan data.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame containing only eligible companies based on thresholds

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> updated_climate_data = lp._apply_production_thresholds(ac._loans)
        """
        if ("turnover" in loan_data.columns) and ("total_assets" in loan_data.columns):
            companies = self._determine_ratios(loan_data)

            eligible_companies = set(
                companies[
                    companies[["turnover_ratio", "asset_ratio"]].isna().sum(axis=1) == 2
                ]["company_id"]
            )

            for sector, threshold in self._settings["production_thresholds"].items():
                eligible_companies.update(
                    companies[
                        (companies["sector"] == sector)
                        & (companies["asset_ratio"] > threshold["asset_ratio"])
                    ]["company_id"]
                )
                eligible_companies.update(
                    companies[
                        (companies["sector"] == sector)
                        & (companies["turnover_ratio"] > threshold["turnover_ratio"])
                    ]["company_id"]
                )

            return loan_data.loc[loan_data["company_id"].isin(eligible_companies), :]
        else:
            return loan_data

    def _determine_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Determine the asset_ratio and the trunover_ratio for the loans present in data

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing loan data.

        Returns
        -------
        pandas.DataFrame
            the data with the asset_ratio and the turnover_ratio added to it
        """
        companies = data.groupby(
            ["technology", "region", "year", "company_id", "sector"], as_index=False
        ).first()
        companies = companies.groupby(["company_id", "sector"], as_index=False).agg(
            {"total_assets": "mean", "turnover": "mean", "production": "sum"}
        )
        companies["asset_ratio"] = companies["production"] / companies["total_assets"]
        companies["turnover_ratio"] = companies["production"] / companies["turnover"]

        return companies

    def _split_loans_over_sector(
        self, loans: pd.DataFrame, loan_indicator: str, year: int
    ) -> pd.DataFrame:
        """
        Split loan amounts over different sectors based on production ratios.

        Parameters:
        -----------
        loans : pd.DataFrame
            DataFrame containing loan data.

        loan_indicator : str
            The loan indicator column name used for splitting loans.

        year : int
            The year to consider for production ratios.

        Returns:
        --------
        pd.DataFrame
            DataFrame with loan amounts split over different sectors.
        """
        # Determine the production in each sector
        climate = self._climate_company_indicators[year].copy()
        climate = climate[
            climate["company_id"].isin(
                self._get_parent_companies(False, year)["parent_company_id"]
            )
        ]
        climate = climate.groupby(["company_id", "sector"], as_index=False)[
            "production"
        ].sum()

        # get the ratio of the production in different sectors to split loan amounts
        climate = (
            self._climate_company_indicators[year]
            .groupby(["company_id", "sector"], as_index=False)["production"]
            .sum()
            .merge(
                climate.groupby(["sector"])["production"].sum(),
                left_on=["sector"],
                right_index=True,
                suffixes=["", "_total"],
            )
        )
        climate["production_fraction"] = climate["production"] / climate["production_total"]

        climate = climate[["company_id", "sector", "production"]].merge(
            climate.groupby(["company_id"])["production"].sum(),
            left_on=["company_id"],
            right_index=True,
            suffixes=["", "_total"],
        )
        climate["ratio"] = climate["production"] / climate["production_total"]

        loans = loans.merge(
            climate[["company_id", "sector", "ratio"]],
            left_on=["company_id", "sector"],
            right_on=["company_id", "sector"],
        )

        # split loan amounts
        loans[loan_indicator] = loans[loan_indicator] * loans["ratio"]
        loans = loans.drop(columns=["ratio"])

        return loans

    def _split_over_technology(
        self, climate_data: pd.DataFrame, loan_indicator: str
    ) -> pd.DataFrame:
        """
        Split loan amounts over different technologies within a sector
        based on production ratios.

        Parameters:
        -----------
        loans : pd.DataFrame
            DataFrame containing loan data.

        loan_indicator : str
            The loan indicator column name used for splitting loans.

        year : int
            The year to consider for production ratios.

        Returns:
        --------
        pd.DataFrame
            DataFrame with loan amounts split over different technologies within
            the same sector.
        """

        grouper_cols = [
            "sector",
            "year",
            self._portfolio_id,
            "company_id",
            "loan_id",
            "portfolio_date",
        ]
        climate_data["production_plus_target"] = (
            climate_data["target"].fillna(0) + climate_data["production"].fillna(0) + 0.0001
        )
        sector_amount = climate_data.groupby(grouper_cols, as_index=False, dropna=False)[
            "production_plus_target"
        ].sum()
        climate_data = climate_data.merge(
            sector_amount,
            how="left",
            left_on=grouper_cols,
            right_on=grouper_cols,
            suffixes=("", "_total"),
        )

        # Perform weighted calculations
        climate_data["technology_ratio"] = (
            climate_data["production_plus_target"]
            / climate_data["production_plus_target_total"]
        )
        climate_data[loan_indicator] = climate_data[loan_indicator] * (
            climate_data["technology_ratio"]
        )

        climate_data = climate_data.drop(
            columns=[
                "production_plus_target",
                "production_plus_target_total",
                "technology_ratio",
            ]
        )

        return climate_data

    def _normalise_production(
        self,
        method: str,
        df_combined: pd.DataFrame,
        year: int,
        economic_weights: dict,
        loan_indicator: str,
    ) -> pd.DataFrame:
        """
        Normalise production data in the climate DataFrame at different levels.

        This method applies different normalisation methods to the combined DataFrame
        containing production data and loan data.

        Parameters
        ----------
        method : str
            Method of normalisation to apply:
            - "global": Perform global normalisation based on sector-wise production totals.
            - "company": Perform company-level normalisation based on sector-wise production totals.
            - "portfolio": Perform normalisation based on portfolio data.
            - "total": Perform normalisation based on combined portfolio data.
            - "economic": Perform normalisation using economic weights.

        df_combined : pd.DataFrame
            The combined DataFrame containing production data and loan data.

        year : int
            The year for which normalisation is applied.

        economic_weights : dict
            Dictionary containing economic weights for different sectors.
            This parameter is used if `method` is set to "economic".

        loan_indicator : str
            column with the loan amount relevant for the analysis.

        Returns
        -------
        pd.DataFrame
            Normalised climate DataFrame based on the specified method.

        """

        if method == "global":
            df_combined = self._global_normalisation(df_combined, year)
        elif method == "economic":
            df_combined = self._economic_normalisation(
                df_combined, economic_weights, year
            )
        elif method == "total":
            df_combined = self._total_normalisation(df_combined, loan_indicator, year)
        elif method == "portfolio":
            df_combined = self._portfolio_normalisation(
                df_combined, loan_indicator, year
            )
        elif method == "company":
            df_combined = self._company_normalisation(df_combined)
        else:
            df_combined["norm"] = 1

        return df_combined

    def _global_normalisation(
        self, df_combined: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """
        Perform global normalisation on combined DataFrame for a specific year.

        This method performs global normalisation on the combined DataFrame for
        a specific year based on sector-wise production totals. These sectors are
        normalised such that the aggregate production of each sector will be equal.

        Parameters
        ----------
        df_combined : pandas.DataFrame
            Combined DataFrame containing loan and climate data.
        year : int
            The year for which global normalisation is being performed.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a norm column added which includes the normalisation weights.
        """
        df_total = self._climate_company_indicators[year].copy()
        companies = self._get_parent_companies(False, year)
        parents = companies["parent_company_id"].unique().tolist()
        df_total = df_total[df_total["company_id"].isin(parents)]
        df_total = df_total.groupby("sector")[["production"]].sum()
        df_combined = df_combined.merge(
            df_total,
            left_on="sector",
            right_on="sector",
            suffixes=("", "_sector_total"),
        )
        df_combined["norm"] = df_combined["production_sector_total"]

        return df_combined.drop(columns=["production_sector_total"])

    def _economic_normalisation(
        self, df_combined: pd.DataFrame, economic_weights: dict, year: int
    ) -> pd.DataFrame:
        """
        Perform economic sector based normalisation on combined DataFrame for
        a specific year with specific weights.

        This method performs an economic sector based normalisation on the combined
        DataFrame for a specific year based on the economic_weights given.

        Parameters
        ----------
        df_combined : pandas.DataFrame
            Combined DataFrame containing loan and climate data.
        economic_weights : str
            The weight of each sector
        year : int
            The year for which global normalisation is being performed.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a norm column added which includes the normalisation weights.
        """
        if economic_weights is None:
            economic_weights = self._settings["economic_weights"]
        df_total = self._climate_company_indicators[year].copy()
        companies = self._get_parent_companies(False, year)
        parents = companies["parent_company_id"].unique().tolist()
        df_total = df_total[df_total["company_id"].isin(parents)]
        df_total = df_total.groupby("sector", as_index=False)[["production"]].sum()
        for sector, weight in economic_weights.items():
            df_total.loc[df_total["sector"] == sector, "production"] = (
                weight * df_total.loc[df_total["sector"] == sector, "production"]
            )
        df_combined = df_combined.merge(
            df_total,
            left_on="sector",
            right_on="sector",
            suffixes=("", "_sector_total"),
        )
        df_combined["norm"] = df_combined["production_sector_total"]

        return df_combined.drop(columns=["production_sector_total"])

    def _total_normalisation(
        self, df_combined: pd.DataFrame, loan_indicator: str, year: int
    ) -> pd.DataFrame:
        """
        Perform combined portfolio based normalisation on combined DataFrame for a specific year.

        This method performs combined portfolio based normalisation on the combined DataFrame
        for a specific year based on sector-wise production totals for companies present
        in the combined portfolio consisting of all the portfolios present in the data. These
        sectors are normalised such that the aggregate production of each sector will be equal
        (if only companies within all the portfolios in the data are considered).

        Parameters
        ----------
        df_combined : pandas.DataFrame
            Combined DataFrame containing loan and climate data.
        loan_indicator : str
            column with the loan amount relevant for the analysis.
        year : int
            The year for which global normalisation is being performed.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a norm column added which includes the normalisation weights.
        """
        df_total = df_combined[df_combined["year"] == year]
        df_total = df_total[df_total[loan_indicator] > 0]
        df_total = df_total.groupby("sector")[["production"]].sum()
        df_combined = df_combined.merge(
            df_total,
            left_on="sector",
            right_on="sector",
            suffixes=("", "_sector_total"),
        )
        df_combined["norm"] = df_combined["production_sector_total"].fillna(1)

        return df_combined.drop(columns=["production_sector_total"])

    def _portfolio_normalisation(
        self, df_combined: pd.DataFrame, loan_indicator: str, year: int
    ) -> pd.DataFrame:
        """
        Perform portfolio based normalisation on combined DataFrame for a specific year.

        This method performs portfolio based normalisation on the combined DataFrame
        for a specific year based on sector-wise production totals for companies present
        in the portfolio. These sectors are normalised such that the aggregate
        production of each sector will be equal (if only companies within the portfolio
        are considered). The normalisation factors could differ for the same company,
        depending on the portfolio it belongs to.

        Parameters
        ----------
        df_combined : pandas.DataFrame
            Combined DataFrame containing loan and climate data.
        loan_indicator : str
            column with the loan amount relevant for the analysis.
        year : int
            The year for which global normalisation is being performed.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a norm column added which includes the normalisation weights.
        """
        df_total = df_combined[df_combined["year"] == year]
        df_total = df_total[df_total[loan_indicator] > 0]
        df_total = df_total.groupby(["sector", self._portfolio_id])[
            ["production"]
        ].sum()
        df_combined = df_combined.merge(
            df_total,
            left_on=["sector", self._portfolio_id],
            right_on=["sector", self._portfolio_id],
            suffixes=("", "_sector_total"),
        )
        df_combined["norm"] = df_combined["production_sector_total"].fillna(1)

        return df_combined.drop(columns=["production_sector_total"])

    def _company_normalisation(
        self,
        df_combined: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform company normalisation on combined DataFrame for a specific year.

        This method performs company normalisation on the combined DataFrame. The
        production of each company in each sector will be normalised, so they are
        equally weighted.

        Parameters
        ----------
        df_combined : pandas.DataFrame
            Combined DataFrame containing loan and climate data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a norm column added which includes the normalisation weights.
        """

        df_total = df_combined.groupby(["sector", "company_id"])[["production"]].sum()
        df_combined = df_combined.merge(
            df_total,
            left_on=["sector", "company_id"],
            right_on=["sector", "company_id"],
            suffixes=("", "_sector_total"),
        )
        df_combined["norm"] = df_combined["production_sector_total"].fillna(1)

        return df_combined.drop(columns=["production_sector_total"])

    def _calculate_alignment_instance(
        self,
        data: pd.DataFrame,
        facet_col: List[str],
        loan_indicator: str = "outstanding_amount",
        bopo_split: bool = False,
        limit: int = 3,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """
        Calculate aggregated results based on input DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame containing data for alignment calculations.

        facet_col : list of str
            List of columns to be used for facet-based aggregation.

        loan_indicator : str, optional
            The loan indicator to use for weighted calculations.
            Default is 'outstanding_amount'.

        bopo_split : bool, optional
            Flag to enable BoPo (Build-Out Phase-Out) split calculations.
            Default is False.

        limit : int, optional
            The limit value used to clip the computed scores.
            Default is 3.

        horizon : int, optional
            The number of years into the future for alignment calculation.
            Default is 5.

        Returns:
        --------
        pd.DataFrame
            Aggregated results DataFrame containing alignment scores.
        """
        cols = set(
            [
                "sector",
                "technology",
                "region",
                self._portfolio_id,
                "company_id",
                "loan_id",
                "portfolio_date",
            ]
            + facet_col
        )

        if "year" in facet_col:
            cols.remove("year")

        results_cols = set(
            ["end_year", "portfolio_date"] + facet_col + [self._portfolio_id]
        )

        if bopo_split:
            results_cols.add("direction")

        cols = list(cols)
        results_cols = list(results_cols)

        end_year = (data["portfolio_date"] / 100).astype(int).max() + horizon

        year_results = data.copy()
        year_results["end_year"] = year_results["year"]

        year_results["deviation"] = year_results["production"] - year_results["target"]
        year_results = self._make_weighted_target(
            year_results, loan_indicator, cols, end_year
        )
        year_results["end_year"] = end_year
        
        if bopo_split:
            year_results = self._make_bopo_split(year_results)

        if "year" not in results_cols:
            year_results = year_results[year_results["year"] == end_year]

        return self._aggregate_results(
            year_results, results_cols, loan_indicator, limit
        )

    def _make_weighted_target(
        self, data: pd.DataFrame, loan_indicator: str, cols: list, end_year: int
    ) -> pd.DataFrame:
        """
        Calculate weighted target values based on loan indicators and specified columns.

        This method computes a weighted deviation and target for the given data.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame containing the data.

        loan_indicator : str
            The loan indicator to use for calculations.

        cols : list of str
            The list of columns to use for grouping.

        end_year : int
            The end year to consider for aggregation.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added columns for weighted deviation and weighted target.
        """
        for sector, sector_settings in self._settings["sectoral_approach"].items():
            for tech in sector_settings["phase_out"] + sector_settings["other"]:
                indexes = (data["technology"] == tech) & (data["sector"] == sector)
                data.loc[indexes, "deviation"] = -1 * data.loc[indexes, "deviation"]  # type: ignore

        data["weighted_deviation"] = data["deviation"] * data[loan_indicator]
        results_target = (
            data[data["end_year"] == end_year]
            .groupby(cols, as_index=False)["target"]
            .first()
        )
        data = data.merge(
            results_target,
            how="left",
            left_on=cols,
            right_on=cols,
            suffixes=("", "_end"),
        )
        data["weighted_target"] = data["target_end"] * data[loan_indicator]

        return data

    def _make_bopo_split(self, data):
        """
        Apply the build-out phase-out split based on sectoral approach settings.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame containing data to be processed.

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with a new 'direction' column indicating the phase-out or build-out direction.
        """

        data["direction"] = "no_change"
        for sector, sector_settings in self._settings["sectoral_approach"].items():
            for tech in sector_settings["phase_out"]:
                indexes = (data["technology"] == tech) & (data["sector"] == sector)
                data.loc[indexes, "direction"] = "phase_out"
            for tech in sector_settings["build_out"]:
                indexes = (data["technology"] == tech) & (data["sector"] == sector)
                data.loc[indexes, "direction"] = "build_out"

        return data

    def _aggregate_results(
        self, data: pd.DataFrame, results_cols: list, loan_indicator: str, limit: int
    ) -> pd.DataFrame:
        """
        Aggregate results based on specified columns and calculate scores (net
        alignment percentages).

        This method aggregates data based on specified columns, computes weighted
        deviation, weighted target, and calculates scores using the loan indicator
        and specified limits.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame containing the data to be aggregated.

        results_cols : list of str
            The list of columns to use for grouping and aggregating.

        loan_indicator : str
            The loan indicator used for weighted calculations.

        limit : int
            The limit value used to clip the computed scores.

        Returns:
        --------
        pd.DataFrame
            DataFrame with aggregated results and computed scores.
        """
        data = data.groupby(results_cols, as_index=False, dropna=False)[
            [loan_indicator, "weighted_deviation", "weighted_target"]
        ].sum()
        data["score"] = data["weighted_deviation"] / data["weighted_target"]
        data["score"] = data["score"].clip(lower=-1 * limit, upper=limit)

        return data

    def _make_portfolio_dates(self) -> set:
        """
        Extracts unique reference dates (`portfolio_date`) from loan data that end with "12".

        Returns:
        --------
        set
            A set containing unique reference dates extracted from loan data,
            filtered based on the last two characters of the `portfolio_date` field.
        """
        if self._loans is None:
            raise ValueError("No loan data provided")

        portfolio_dates = set(
            self._loans[self._loans["portfolio_date"].astype(str).str[-2:] == "12"][
                "portfolio_date"
            ].unique()
        )
        return portfolio_dates

    def _make_master_data(
        self,
        loan_indicator: str,
        only_parents: bool,
        facet_col: list,
        use_region_file: bool,
        normalise_method: str,
    ) -> dict:
        """
        Generate master data dictionary containing preprocessed data for specified scenarios.

        This method iterates over scenario years and data years to preprocess data for valid combinations
        based on the presence of scenario sets and pathways in the scenario data.

        Parameters:
        -----------
        loan_indicator : str
            column with the loan amount relevant for the analysis.

        only_parents : bool
            Flag indicating whether to include only parent companies.

        facet_col : list
            List of columns used as facets for the data preprocessing.

        use_region_file : bool
            Flag indicating whether to use region file data for preprocessing.

        normalise_method : str
            Method of normalisation to apply:
            - "global": Perform global normalisation based on sector-wise production totals.
            - "company": Perform company-level normalisation based on sector-wise production totals.
            - "portfolio": Perform normalisation based on portfolio data.
            - "total": Perform normalisation based on the combined portfolio data.
            - "economic": Perform normalisation using economic weights.

        Returns:
        --------
        dict
            A dictionary containing preprocessed data for each scenario year and associated data years.
            The dictionary structure is:
            {
                scenario_year_1: {
                    data_year_1: preprocessed_data_1,
                    data_year_2: preprocessed_data_2,
                    ...
                },
                scenario_year_2: {
                    data_year_1: preprocessed_data_1,
                    data_year_2: preprocessed_data_2,
                    ...
                },
                ...
            }
            where `preprocessed_data` is a DataFrame containing processed data for a specific year combination.
        """

        master_data = {}
        for scenario_year in self._scenario_data.keys():
            if (self._scenario_set in self._scenario_data[scenario_year]) and (
                self._pathway in self._scenario_data[scenario_year][self._scenario_set]
            ):
                master_data_year = {}
                for data_year in self._climate_company_indicators.keys():
                    if (self._scenario_set in self._scenario_data[data_year]) and (
                        self._pathway
                        in self._scenario_data[data_year][self._scenario_set]
                    ):
                        if -1 < (data_year - scenario_year) < 2:
                            master_data_year[data_year] = self._preprocess_data(
                                True,
                                False,
                                loan_indicator,
                                only_parents,
                                facet_col,
                                use_region_file,
                                data_year,
                                normalise_method,
                                scenario_year,
                            )
                master_data[scenario_year] = master_data_year

        return master_data

    def _aggregate_over_time_results(
        self, data: pd.DataFrame, loan_indicator: str, add_total: bool
    ) -> pd.DataFrame:
        """
        Aggregate the results from an over time analysis.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the results from
            calculate_net_alignment_change_over_time.

        loan_indicator : str
            column with the loan amount relevant for the analysis.

        add_total : bool
            Flag indicating whether to add a total column to the results.

        Returns
        -------
        pd.DataFrame
            Aggregated and analysed DataFrame based on the specified conditions.
        """

        results = []
        if add_total:
            total_group = data.groupby(
                ["scenario_year", "data_year", "portfolio_date", "end_year"]
            ).sum(numeric_only=True)
            total_group[self._portfolio_id] = "total"
            total_group["score"] = (
                total_group["weighted_deviation"] / total_group["weighted_target"]
            )
            data = pd.concat(
                [
                    data,
                    total_group.drop(columns=["weighted_deviation", "weighted_target"]),
                ]
            )
        for portfolio_id in data[self._portfolio_id].unique():
            data_portfolio = data[data[self._portfolio_id] == portfolio_id].reset_index(
                drop=True
            )
            data_portfolio = data_portfolio.sort_values(
                by=["scenario_year", "data_year", "portfolio_date"]
            )
            if (3 * data_portfolio["scenario_year"].diff().mean() == 1) & (
                data_portfolio[loan_indicator].sum() > 0
            ):
                data_portfolio, scores = self._make_time_metrics(data_portfolio)
                if (
                    ("decarbonisation_shift" in data_portfolio.columns)
                    & ("portfolio_shift" in data_portfolio.columns)
                    & ("counterparty_shift" in data_portfolio.columns)
                ):
                    data_portfolio["total_shift"] = (
                        data_portfolio["decarbonisation_shift"]
                        + data_portfolio["portfolio_shift"]
                        + data_portfolio["counterparty_shift"]
                    )
                    results.append(
                        data_portfolio.merge(
                            scores,
                            how="right",
                            left_on=[self._portfolio_id, "data_year"],
                            right_on=[self._portfolio_id, "data_year"],
                        ).drop(columns=["data_year"])
                    )
        return pd.concat(results)

    def _make_time_metrics(self, data_portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates the decarbonisation shift, portfoliio shift and counterparty shift
        from the results calculated with the calculate_net_alignment_change_over_time
        method.

        Parameters
        ----------
        data_portfolio : pd.DataFrame
            The DataFrame containing the results from
            calculate_net_alignment_change_over_time for a specific portfolio.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            data_portfolio with the addition of the decarbonisation_shift,
            portfoliio_shift and counterparty_shift columns and a dataframe with
            the original scores
        """

        data_portfolio = (
            data_portfolio[["score"]]
            .diff()
            .rename(columns={"score": "difference"})
            .merge(data_portfolio, left_index=True, right_index=True)
        )
        data_portfolio["valid_rows"] = (
            (data_portfolio["scenario_year"] == (data_portfolio["end_year"] - 5))
            & (data_portfolio["scenario_year"] == data_portfolio["data_year"])
            & (
                data_portfolio["scenario_year"]
                == (data_portfolio["portfolio_date"] / 100).astype(int)
            )
        )
        scores = data_portfolio[data_portfolio["valid_rows"]][
            [self._portfolio_id, "portfolio_date", "data_year", "score"]
        ]
        data_portfolio["metric"] = ""
        data_portfolio.loc[
            data_portfolio["valid_rows"].shift(-0).fillna(False), "metric"
        ] = "decarbonisation_shift"
        data_portfolio.loc[
            data_portfolio["valid_rows"].shift(-1).fillna(False), "metric"
        ] = "portfolio_shift"
        data_portfolio.loc[
            data_portfolio["valid_rows"].shift(-2).fillna(False), "metric"
        ] = "counterparty_shift"
        data_portfolio = (
            data_portfolio[[self._portfolio_id, "data_year", "metric", "difference"]]
            .dropna()
            .pivot_table(
                index=[self._portfolio_id, "data_year"],
                columns="metric",
                values="difference",
            )
            .reset_index(drop=False)
        )

        return data_portfolio, scores
