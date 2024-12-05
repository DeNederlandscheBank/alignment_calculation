from typing import Any
import pandas as pd
from unicodedata import normalize as unicode_norm
from .load_climate_data import _load_main_climate_data
from .read_data import _load_loan_counterparties, _load_loan_data, _add_external_data

from .ac_config import alignmentCalculatorConfig


class loanbookPreparer:
    """
    The loanbookPreparer class can construct a loanbook based on loan data, possible
    augmented with external data and connect it to the climate dataset. This can yield
    a loanbook that can directly be used in the alignment calculator.

    Parameters
    ----------
    climate_file_location: list | str | None, optional
        the location of the climate file or a list of climate files
        default = None

    settings: dict, optional
        the settings equivalent to the settings from the parameters.py file
        that should apply when loading the data. If None the settings
        from parameters.py will be used.
        default = None
    """

    def __init__(
        self,
        climate_file_location: list | str | None = None,
        settings: dict | None = None,
        remove_government: bool = True,
    ):
        self._external_columns = {}
        self._load_climate_files(climate_file_location, settings)
        self._preprocess_climate(remove_government)

    def _load_climate_files(
        self, climate_file_location: list | str | None, settings: dict | None
    ) -> None:
        """
        loads the climate data and sets it to the _climate attribute

        Parameters:
        -----------
        climate_file_location: list | str | None
            the location of the climate file or a list of climate files

        settings: dict | None
            the settings equivalent to the settings from the parameters.py file
            that should apply when loading the data. If None the settings
            from parameters.py will be used.
        """

        climate_files = []
        if climate_file_location is None:
            if settings is None:
                climate_file_names = (
                    alignmentCalculatorConfig()
                    .load_settings()["main_climate_file"]
                    .values()
                )
            else:
                climate_file_names = settings["main_climate_file"].values()
            for climate_file in climate_file_names:
                climate_files.append(
                    _load_main_climate_data(climate_file)["company_indicators"]
                )
        elif isinstance(climate_file_location, list):
            for climate_file in climate_file_location:
                climate_files.append(
                    _load_main_climate_data(climate_file)["company_indicators"]
                )
        else:
            climate_files = [
                _load_main_climate_data(climate_file_location)["company_indicators"]
            ]
        self._climate = pd.concat(climate_files)

    def prepare_loanbook(
        self,
        base_year: int = 2023,
        month: int = 12,
        start_year: int | None = None,
        start_month: int | None = None,
        match_data: str | None = None,
        portfolio_codes: list | None = None,
        loanbook_filename: str | None = None,
        frequency: str | None = None,
        additional_columns: dict | None = None,
        external_columns: dict | None = None,
        loan_counterparties_args: dict[str,Any]|None = None,
    ) -> pd.DataFrame | None:
        """
        Prepares a loanbook based on loan data for a specified year and month.

        Parameters:
        -----------
        base_year : int, optional
            The year for which loan data will be loaded.
            Default = 2022.

        month : int, optional
            The month for which loan data will be loaded.
            Default = 12.

        start_year : int|None, optional
            The year from which to start loading the loan data.
            Default = None.

        start_month : int|None, optional
            The month from which to start loading the loan data.
            Default = None.

        match_data: str|None, optional
            The location of the matched data file, containing the ids of the
            climate companies (company_id) and the counterpaty ids (counterparty_id)
            in a csv format.
            Default = None.

        portfolio_codes : list|None, optional
            List of portfolio codes to filter the loan data.
            Default = None.

        loanbook_filename : str|None, optional
            If provided, the loanbook data will be saved as a CSV file with the
            specified filename. If not provided, the loanbook DataFrame will be
            returned without saving.
            default = None

        frequency: str, optional
            String indicating what the frequency of the fetched data should be,
            - 'M': Monthly
            - 'Q': Quarterly
            - 'Y': Yearly
            Default = None

        additional_columns: dict|None, optional
            Additional columns that should be loaded from the loan data.
            Default = None.

        external_columns : dict|None, optional
            The extra columns for which data should be fetched. The value relates to the
            expression that should be added to the query to get the additional data and
            the key relates to how the additional datapoint should be called.
            Default = None.
            
        loan_counterparties_args: dict[str,Any]|None, optional
            The arguments for the load loan counterparties function
            Default = None.

        Returns:
        --------
        pandas.DataFrame | None
            If loanbook_filename is None, returns the loanbook as a pandas DataFrame.
            If loanbook_filename is provided, saves the loanbook as a CSV file and
            returns None.
        """

        if additional_columns is None:
            self._additional_columns = dict()
        else:
            self._additional_columns = additional_columns

        if portfolio_codes is None:
            portfolio_codes = []

        if external_columns is None:
            self._external_columns = dict()
        else:
            self._external_columns = external_columns

        if loan_counterparties_args is None:
            loan_counterparties_args = {}

        loan_counterparties = _load_loan_counterparties(**loan_counterparties_args)

        if match_data is None:
            climate_data = self._match_data(loan_counterparties)
        else:
            loan_counterparties = pd.read_csv(match_data)
            climate_data = self._simple_join(loan_counterparties)

        if len(self._external_columns) > 0:
            climate_data = _add_external_data(climate_data, self._external_columns)
            
        loan_data_args = {}
        if frequency is not None:
            loan_data_args["frequency"] = frequency
        if len(self._additional_columns) > 0:
            loan_data_args["additional_columns"] = self._additional_columns
        loan_data = _load_loan_data(
            climate_data=climate_data,
            year=base_year,
            month=month,
            portfolio_codes=portfolio_codes,
            start_month=start_month,
            start_year=start_year,
            **loan_data_args
        )
        
        loanbook = self._merge_climate_loan_data(climate_data, loan_data)
        loanbook = self._post_processed(loanbook)

        if loanbook_filename is None:
            return loanbook
        else:
            loanbook.to_csv(loanbook_filename)

    def _remove_cities_and_states(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes city, provices, states and governments from the data based on the
        name_company column

        Parameters:
        -----------

        data: pandas.DataFrame
            The dataframe from which the governemental bodies should be removed.

        Returns:
        --------
        pandas.DataFrame
            The processed dataframe without the governemental bodies.
        """
        data = data[~data["name_company"].str.lower().str.contains("government of ")]
        data = data[~data["name_company"].str.lower().str.contains("city of ")]
        data = data[~data["name_company"].str.lower().str.contains("province of ")]
        data = data[~data["name_company"].str.lower().str.contains("state of ")]

        return data

    def _merge_climate_loan_data(
        self, climate_data: pd.DataFrame, loan_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        merges the loan and climate datasets.

        Parameters:
        -----------

        climate_data: pandas.DataFrame
            The fully processed climate data

        loan_data: pandas.DataFrame
            The loan data.

        Returns:
        --------
        pandas.DataFrame
            the combined loan climate loanbook.
        """
        loanbook = climate_data.merge(
            loan_data,
            how="inner",
            left_on=["counterparty_id"],
            right_on=["counterparty_id"],
        )

        return loanbook

    def _process_name(self, names: pd.Series) -> pd.Series:
        """
        Processes the names to remove punctuations, non-unicode characters,
        makes it lowercase, duplicate spaces and fixes two common abbriviations
        """
        names = names.apply(
            lambda string: unicode_norm("NFKD", str(string))
            .encode("ASCII", "ignore")
            .decode()
        )
        names = names.str.lower()
        names = names.str.replace(r"[^\w\s]", "", regex=True)
        names = names.str.replace("  ", " ")
        names = names.str.replace("company", "co")
        names = names.str.replace("corporation", "corp")
        names = names.str.strip()

        return names

    def _preprocess_climate(self, remove_government: bool):
        """
        preprocess the climate data by removing the the governements from the data,
        deduplicating the data and processing the company names.

        Parameters:
        -----------

        remove_government: bool
            Flag whether to remove governments from the climate data
        """
        if remove_government:
            self._climate = self._remove_cities_and_states(self._climate)
        self._climate = (
            self._climate[["company_id", "name_company", "lei"]]
            .sort_values(by=["lei"], ascending=False)
            .drop_duplicates(subset=["company_id"], keep="first")
        )
        self._climate["processed_name"] = self._process_name(self._climate["name_company"])

    def _preprocessed_names(self, loan_data: pd.DataFrame):
        """
        Generates the processed names for the loan data for both the company
        name as well as the parent company name.

        Parameters:
        -----------
        loan_data: pd.DataFrame
            The loan data that should be merged with the climate data
        """

        loan_data["processed_name"] = self._process_name(loan_data["company_name"])
        loan_data["processed_parent_name"] = self._process_name(
            loan_data["parent_name"]
        )

    def _join(self, loan_data: pd.DataFrame) -> pd.DataFrame:
        """
        joins the loan data to the climate data, first direct on lei, second on
        parent lei, third on name, fourth on parent name

        Parameters:
        -----------
        loan_data: pd.DataFrame
            The loan data that should be merged with the climate data

        Returns:
        --------
        pandas.DataFrame
            the combined climate loan dataframe
        """

        direct = loan_data.merge(
            self._climate.dropna(subset=["lei"]),
            how="inner",
            left_on="company_lei",
            right_on="lei",
        )
        parent = loan_data.merge(
            self._climate.dropna(subset=["lei"]),
            how="inner",
            left_on="parent_lei",
            right_on="lei",
        )
        direct_name = loan_data.merge(
            self._climate,
            how="inner",
            left_on="processed_name",
            right_on="processed_name",
        )
        parent_name = loan_data.merge(
            self._climate,
            how="inner",
            left_on="processed_parent_name",
            right_on="processed_name",
        )

        combined = pd.concat([direct, direct_name, parent, parent_name])

        combined = self._postprocess_join(combined)

        return combined

    def _postprocess_join(self, combined: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess the data after the climate data has been joined with the
        counterparty id data.

        Parameters:
        -----------

        combined: pandas.DataFrame
            The combined counterparty - climate dataset

        Returns:
        --------
        pandas.DataFrame
            The dataset, without duplicates and with the columns name_company,
            company_country, company_lei, parent_name, parent_lei add if they did not exist
        """
        group_columns = [
            "company_name",
            "name_company",
            "company_country",
            "company_lei",
            "parent_name",
            "parent_lei",
        ]

        if len(self._external_columns) > 0:
            group_columns.append("external_id")

        for column in group_columns:
            if column not in combined.columns:
                combined[column] = ""
        combined = combined.groupby(["counterparty_id", "company_id"], as_index=False)[
            group_columns
        ].first()

        return combined

    def _simple_join(self, matching_ids: pd.DataFrame) -> pd.DataFrame:
        """
        Join a matching table to the climate data

        Parameters:
        -----------

        matching_ids: pandas.DataFrame
            A pandas dataframe with at least two columns, a column with the climate_data_id,
            company_id, and with the counterparty_id from a loanbook, counterparty_id.

        Returns:
        --------
        pandas.DataFrame
            The climate dataset with counterparty ids added to the data and the name_company,
            company_country, company_lei, parent_name, parent_lei columns if they did not exist
        """

        combined = self._climate.merge(
            matching_ids, how="inner", left_on=["company_id"], right_on=["company_id"]
        )
        combined = self._postprocess_join(combined)

        return combined

    def _post_processed(self, combined: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocesses the combined loanbook.

        Parameters:
        -----------

        combined: pandas.DataFrame
            The combined climate-loan dataframe

        Returns:
        --------
        pandas.DataFrame
            the postprocessed loanbook as a pandas DataFrame.
        """

        combined = combined.drop_duplicates(
            subset=["portfolio_code", "counterparty_id", "loan_id", "portfolio_date"]
        )
        combined = combined.reset_index(drop=True)
        valid_columns = (
            [
                "portfolio_code",
                "portfolio_date",
                "counterparty_id",
                "company_name",
                "name_company",
                "company_country",
                "company_lei",
                "parent_name",
                "parent_lei",
                "outstanding_amount",
                "loan_id",
                "company_id",
            ]
            + list(self._additional_columns.keys())
            + list(self._external_columns.keys())
        )
        if len(self._external_columns) > 0:
            valid_columns.append("external_id")

        combined = combined[valid_columns]

        return combined

    def _match_data(self, loan_data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the steps to match the loan data to the climate data.

        Parameters:
        -----------
        loan_data: pd.DataFrame
            The loan data that should be merged with the climate data

        Returns:
        --------
        pandas.DataFrame
            the fully processed loanbook as a pandas DataFrame.
        """

        self._preprocessed_names(loan_data)
        loanbook = self._join(loan_data)

        return loanbook
