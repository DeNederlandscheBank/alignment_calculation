from typing import Union
import pandas as pd
import os


def _load_loan_data(
    climate_data: pd.DataFrame,
    year: int,
    month: int,
    portfolio_codes: list,
    start_month: int | None,
    start_year: int | None,
    data_file: str = "../data/loan_data/loan_data.csv",
) -> pd.DataFrame:
    """
    Load loan data based on the counterparty_id for the combined climate data.

    Parameters:
    -----------
    climate_data : pd.DataFrame
        DataFrame containing the combined climate data.
    year : int
        The year for loan data.
    month : int
        The month for loan data.
    portfolio_codes : list
        List of portfolio codes for filtering loan data.
    start_month : int | None
        The start month for loan data.
    start_year : int | None
        The start year for loan data.
    data_file: str
        The data file containing the counterparty data
        default='../data/loan_data/loan_data.csv'

    Returns:
    --------
    pd.DataFrame
        loan data
    """

    absolute_path = data_file.replace("..", os.path.dirname(__file__))
    absolute_path = os.path.normpath(absolute_path)

    single_period = False
    if start_year is None:
        single_period = True
        start_year = year
    if start_month is None:
        start_month = month

    df = pd.read_csv(absolute_path)
    df["loan_id"] = df["loan_id"].astype(str)
    df = df.loc[df["counterparty_id"].isin(climate_data["counterparty_id"])]
    
    if single_period:
        df = df.loc[df["portfolio_date"].astype(str).str.contains(str(start_year))]
        df = df.loc[df["portfolio_date"].astype(str).str.contains(str(start_month))]

    if isinstance(portfolio_codes, str):
        df = df.loc[df["portfolio_code"].isin(portfolio_codes)]

    return df


def _load_loan_counterparties(
    data_file: str = "../data/loan_data/loan_companies.csv",
    year: int = 2023,
    month: int = 12
) -> pd.DataFrame:
    """
    Reads the counterparties from a counterparty file

    Parameters
    ----------
    data_file: str
        The data file containing the counterparty data
        default='../data/loan_data/loan_companies.csv'
    year: int
        The year for which the data should be loaded
        default=2023
    month: int
        The month for which the data should be loaded
        default=12

    Returns
    -------
    pandas.DataFrame
        The counterparty data from the file
    """
    absolute_path = data_file.replace("..", os.path.dirname(__file__))
    absolute_path = os.path.normpath(absolute_path)

    return pd.read_csv(absolute_path)


def _add_external_data(data: pd.DataFrame, columns: dict):
    """
    Adds the columns to the data based on the dict. This can be extended to add data
    from external sources


    Parameters
    ----------
    data: pandas.DataFrame
        The data file to which the external data should be added
    columns: dict
        The columns that should be added to the dataframe, with the key the name of
        the column and the value the arguments needed to add the columns

    Returns
    -------
    pandas.DataFrame
        The data files with the additional external data
    """

    for key, value in columns.items():
        data[key] = value

    return data
