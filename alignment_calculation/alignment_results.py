import pandas as pd


class alignmentResults:

    def __init__(
        self,
        results_data: pd.DataFrame,
        climate_company_indicators: dict,
        df_climate: dict,
        scenario_data: dict,
        settings: dict,
        portfolio_id: str,
    ) -> None:
        """
        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data to group and calculate scores.
        """
        self._results_data = results_data
        self._climate_company_indicators = climate_company_indicators
        self._df_climate = df_climate
        self._scenario_data = scenario_data
        self._settings = settings
        self._portfolio_id = portfolio_id

    def get_results(self) -> pd.DataFrame:
        """
        Get the results from the alignment calculation as a dataframe

        returns
        -------
        pandas.DataFrame
            DataFrame containing results data.
        """
        return self._results_data.drop(
            columns=["weighted_deviation", "weighted_target"]
        )

    def group_scores(self, grouper: list| None = None) -> pd.DataFrame:
        """
        Group scores in the results data based on specified groupers. if no
        grouper is provided the results_data is just returned. The results
        data should have a weighted_deviation and weighted_target column to
        allow for the calculation of the grouped scores. If no grouper is given
        the original scores are returned.

        Parameters
        ----------
        grouper : list of str or None, optional
            List of columns to group the results data by (default is None).
            If provided, the scores will be aggregated based on these groupers.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing grouped scores with calculated 'score' column.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = pd.DataFrame({
        ...     'group': ['A', 'A', 'A', 'B'],
        ...     'weighted_deviation': [10, 20, 30, 40],
        ...     'weighted_target': [100, 200, 300, 200]
        ... })
        >>> ac.group_scores(results, grouper=['group'])
          group  weighted_deviation  weighted_target     score
        0     A                  60              600  0.100000
        1     B                  40              200  0.200000
        """
        if grouper is None:
            return self._results_data
        else:
            results_data = self._results_data.groupby(grouper, as_index=False).sum(
                numeric_only=True
            )

        results_data["score"] = (
            results_data["weighted_deviation"] / results_data["weighted_target"]
        )
        return results_data.drop(columns=["weighted_deviation", "weighted_target"])

    def add_information_to_results(
        self,
        loan_indicator: str,
        main_sector: bool = True,
        production_values: bool = True,
        target_values: bool = True,
        company_names: bool = True,
        company_domicile: bool = True,
        plant_locations: bool = True,
    ) -> pd.DataFrame:
        """
        Add additional information to the results data based on specified options.

        Parameters
        ----------
        loan_indicator : str
            column with the loan amount relevant for the analysis.
        main_sector : bool, optional
            Whether to add main sector information (default is True).
        production_values : bool, optional
            Whether to add production values (default is True).
        target_values : bool, optional
            Whether to add target values (default is True).
        company_names : bool, optional
            Whether to add company names (default is True).
        company_domicile : bool, optional
            Whether to add company domicile information (default is True).
        plant_locations : bool, optional
            Whether to add plant locations (default is True).

        Returns
        -------
        pandas.DataFrame
            DataFrame with additional information added based on specified options.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = pd.DataFrame({
        ...     'company_id': [234, 2355, 31],
        ...     'score': [-0.1, 0.31, -0.4]
        ...     'outstanding_amount': [100241547, 4847964, 11235887]
        ... })
        >>> updated_results = ac.add_information_to_results(
        ...     results,
        ...     loan_indicator='outstanding_amount',
        ...     production_values=False,
        ...     target_values=False,
        ...     plant_locations=False
        ... )
        >>> print(updated_results)
           company_id  score   outstanding_amount   main_sector  company_names  company_domicile
        0      234      -0.1        100241547           ...            ...            ...
        1     2355      0.31        4847964             ...            ...            ...
        2      31       -0.4        11235887            ...            ...            ...
        """
        results_data = self._results_data.copy()
        if main_sector:
            results_data = self._add_main_sector(results_data, loan_indicator)
        if company_names:
            results_data = self._add_company_names(results_data)
        if company_domicile:
            results_data = self._add_company_domicile(results_data)
        if plant_locations:
            results_data = self._add_production_location(results_data, loan_indicator)
        if production_values:
            results_data = self._add_production(results_data)
        if target_values:
            results_data = self._add_target(results_data)

        return results_data

    def _add_company_names(
        self,
        results_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add company names to results data based on scenario data.

        This method merges company names from scenario data with results data
        based on company identifiers. The portfolio_date should be included in the
        results data as this is used to determine from which year of the AI
        data the name will be taken.

        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with company names added to the results data.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = pd.DataFrame({
        ...     'company_id': [1, 2, 3],
        ...     'portfolio_date': ['2022-01-01', '2023-01-01', '2024-01-01']
        ... })
        >>> ac._add_company_names(results)
           company_id    portfolio_date name_company
        0           1  2022-01-01       Company A
        1           2  2023-01-01       Company B
        2           3  2024-01-01       Company C
        """
        data = []
        for year in self._scenario_data.keys():
            if year in self._climate_company_indicators.keys():
                names = self._df_climate[year][
                    ["company_id", "name_company"]
                ].drop_duplicates()
                data.append(
                    results_data[
                        results_data["portfolio_date"]
                        .astype(str)
                        .str.contains(str(year))
                    ].merge(
                        names,
                        how="left",
                        left_on=["company_id"],
                        right_on=["company_id"],
                    )
                )

        return pd.concat(data)

    def _add_main_sector(
        self, results_data: pd.DataFrame, loan_indicator: str
    ) -> pd.DataFrame:
        """
        Add the main sector information to results data based on loan indicators.

        This method identifies the main sector for each company based on the
        loan indicator column and adds this information to the results data.

        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data.
        loan_indicator : str
            column with the loan amount relevant for the analysis.

        Returns
        -------
        pandas.DataFrame
            DataFrame with main sector information added to the results data.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = pd.DataFrame({
        ...     'company_id': [1, 2, 2],
        ...     'sector': ['coal', 'power', 'steel'],
        ...     'outstanding_amount': [100, 200, 300]
        ... })
        >>> ac._add_main_sector(results, loan_indicator='outstanding_amount')
           company_id  sector  outstanding_amount sector_main
        0           1   coal              100        coal
        1           2   power             200        steel
        2           2   steel             300        steel
        """

        sectors = (
            results_data.sort_values(by=loan_indicator)
            .groupby("company_id")["sector"]
            .last()
        )
        results_data = results_data.merge(
            sectors,
            how="left",
            left_on="company_id",
            right_on="company_id",
            suffixes=("", "_main"),
        )

        return results_data

    def _add_production(
        self,
        results_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add production information to results data based on company indicators.

        This method calculates and adds production information to the results data
        based on company indicators and sectoral approaches.

        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with production information added to the results data.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = pd.DataFrame({
        ...     'company_id': [1, 2, 2],
        ...     'sector': ['coal', 'power', 'power'],
        ...     'technology': ['coal', 'oilcap', 'coalcap'],
        ...     'year': [2023, 2023, 2023],
        ... })
        >>> ac._add_production(results)
           company_id  sector technology  year  production
        0           1    coal   coal       2022    1000
        1           2   power   oilcap     2023    2000
        2           2   power   coalcap    2023    3000
        """
        data = []
        for year in self._climate_company_indicators.keys():
            production = (
                self._climate_company_indicators[year]
                .groupby(
                    ["company_id", "sector", "technology", "year"], as_index=False
                )["production"]
                .sum()
            )
            for sector, sector_settings in self._settings["sectoral_approach"].items():
                if sector_settings["approach"] == "sda":
                    production.loc[production["sector"] == sector, "technology"] = (
                        sector
                    )
            production = production.groupby(
                ["company_id", "sector", "technology", "year"], as_index=False
            )["production"].sum()
            data.append(
                results_data[
                    results_data["portfolio_date"].astype(str).str.contains(str(year))
                ]
                .merge(
                    production,
                    how="left",
                    left_on=["company_id", "sector", "technology", "end_year"],
                    right_on=["company_id", "sector", "technology", "year"],
                )
                .drop(columns="year")
            )

        return pd.concat(data)

    def _add_company_domicile(self, results_data: pd.DataFrame):
        """
        Add company domicile information to results data.

        This method reads company domicile information from a CSV file and adds
        it to the results data based on company identifiers.

        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with company domicile information added to the results data.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = pd.DataFrame({
        ...     'company_id': [1, 2, 3],
        ... })
        >>> ac._add_company_domicile(results)
           company_id   domicile
        0           1      DE
        1           2      ES
        2           3      IT
        """
        df = pd.read_csv(self._settings["company_information_file"])
        results_data = results_data.merge(
            df[["company_id", "domicile"]],
            how="left",
            left_on="company_id",
            right_on="company_id",
        )

        return results_data

    def _add_production_location(
        self,
        results_data: pd.DataFrame,
        loan_indicator: str,
        production_fill: float = 0.0001,
        country_fill: str = "XX",
    ) -> pd.DataFrame:
        """
        Add production location information to the results data. If no location is given
        the location is set to country_fill if no production is given the amount of
        production is set to production_fill.

        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data.
        loan_indicator : str
            column with the loan amount relevant for the analysis.
        production_fill: float
            fill value if no production is present in the data
            default=0.0001,
        country_fill: str
            fill value if no country data is given
            default='XX',

        Returns
        -------
        pandas.DataFrame
            DataFrame with production location information added to the results data.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = alignmentCalculator.calculate_net_alignment()
        >>> ac._add_production_location(results, loan_indicator='outstanding_amount')
           company_id  sector  year production
        0        10     power  2023   1231
        1        12     power  2023   9482
        2       320     power  2023    200
        """

        sda_technologies = []
        for sector_setting in self._settings["sectoral_approach"].values():
            if sector_setting["approach"] == "sda":
                sda_technologies = sda_technologies + sector_setting["other"]

        data = []
        for year in self._climate_company_indicators.keys():
            production = self._climate_company_indicators[year]
            production["plant_location"] = production["plant_location"].fillna(
                country_fill
            )
            data_combined = []

            for sector, sector_setting in self._settings["sectoral_approach"].items():
                production_sector = production[production["sector"] == sector].copy()
                sector_data = results_data[results_data["sector"] == sector].copy()
                include_technology = ["technology"]
                if sector_setting["approach"] == "sda":
                    include_technology = []

                production_sector = production_sector.groupby(
                    ["company_id", "sector", "year", "plant_location"]
                    + include_technology,
                    as_index=False,
                )["production"].sum()
                data_plus_production = sector_data[
                    sector_data["portfolio_date"].astype(str).str.contains(str(year))
                ].merge(
                    production_sector,
                    how="left",
                    left_on=["company_id", "sector", "end_year"] + include_technology,
                    right_on=["company_id", "sector", "year"] + include_technology,
                )
                data_plus_production["production"] = (
                    data_plus_production["production"].fillna(0) + production_fill
                )
                production_company_technology = (
                    data_plus_production.groupby(
                        ["company_id", "sector", "end_year", self._portfolio_id]
                        + include_technology,
                        as_index=False,
                    )["production"]
                    .sum()
                    .rename(columns={"production": "production_total"})
                )
                data_combined.append(
                    data_plus_production.merge(
                        production_company_technology,
                        how="left",
                        left_on=["company_id", "sector", "end_year", self._portfolio_id]
                        + include_technology,
                        right_on=[
                            "company_id",
                            "sector",
                            "end_year",
                            self._portfolio_id,
                        ]
                        + include_technology,
                    )
                )

            data_plus_production = pd.concat(data_combined)
            data_plus_production["ratio"] = (
                data_plus_production["production"]
                / data_plus_production["production_total"]
            )
            data_plus_production[loan_indicator] = (
                data_plus_production[loan_indicator] * data_plus_production["ratio"]
            )
            data.append(
                data_plus_production.drop(
                    columns=["ratio", "production", "production_total", "year"]
                )
            )

        return pd.concat(data)

    def _add_target(
        self,
        results_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add target information to results data based on available scenarios.

        This method calculates and adds target information to the results data
        based on available scenarios and company indicators.

        Parameters
        ----------
        results_data : pandas.DataFrame
            DataFrame containing results data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with target information added to the results data.

        Examples
        --------
        >>> ac = alignmentCalculator()
        >>> results = alignmentCalculator.calculate_net_alignment()
        >>> ac._add_production_location(results, loan_indicator='outstanding_amount')
           company_id  sector  year target
        0        10     power  2023   1251
        1        12     power  2023   9210
        2       320     power  2023    352
        """

        data = []
        for year in self._scenario_data.keys():
            if year in self._climate_company_indicators.keys():
                production = (
                    self._df_climate[year]
                    .groupby(
                        ["company_id", "sector", "technology", "year"],
                        as_index=False,
                    )["target"]
                    .sum()
                )
                for sector, sector_settings in self._settings[
                    "sectoral_approach"
                ].items():
                    if sector_settings["approach"] == "sda":
                        production.loc[production["sector"] == sector, "technology"] = (
                            sector
                        )
                production = production.groupby(
                    ["company_id", "sector", "technology", "year"], as_index=False
                )["target"].sum()
                data.append(
                    results_data[
                        results_data["portfolio_date"]
                        .astype(str)
                        .str.contains(str(year))
                    ]
                    .merge(
                        production,
                        how="left",
                        left_on=["company_id", "sector", "technology", "end_year"],
                        right_on=["company_id", "sector", "technology", "year"],
                    )
                    .drop(columns="year")
                )

        return pd.concat(data)
