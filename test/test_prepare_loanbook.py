import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock
from alignment_calculation.prepare_loanbook import loanbookPreparer


@pytest.fixture
def preparer():
    return loanbookPreparer()


@pytest.fixture
def sample_climate_data():
    return pd.DataFrame(
        {
            "company_id": [3, 4],
            "name_company": ["Company A", "Government of B"],
            "lei": ["123", "456"],
        }
    )


@pytest.fixture
def sample_loan_data():
    return pd.DataFrame(
        {
            "counterparty_id": [1, 2],
            "loan_id": [3, 4],
            "portfolio_code": [5, 6],
            "portfolio_date": [2022, 2022],
            "company_name": ["Company A", "Company B"],
            "parent_name": ["Parent A", "Parent B"],
            "company_country": ["US", "DE"],
            "company_lei": ["123", "456"],
            "parent_lei": ["789", "012"],
            "outstanding_amount": [10000000, 900000],
        }
    )


def test_load_climate_files(preparer):
    with patch(
        "alignment_calculation.prepare_loanbook._load_main_climate_data",
        return_value=pd.DataFrame({"company_indicators": []}),
    ) as mock_load:
        preparer._load_climate_files("some_file", None)
        mock_load.assert_called()


def test_remove_cities_and_states(preparer, sample_climate_data):
    expected = pd.DataFrame(
        {"company_id": [3], "name_company": ["Company A"], "lei": ["123"]}
    )
    result = preparer._remove_cities_and_states(sample_climate_data)
    assert_frame_equal(result, expected)


def test_merge_climate_loan_data(preparer, sample_climate_data, sample_loan_data):
    sample_climate_data["counterparty_id"] = [1, 2]
    expected = pd.DataFrame(
        {
            "company_id": [3, 4],
            "name_company": ["Company A", "Government of B"],
            "lei": ["123", "456"],
            "counterparty_id": [1, 2],
            "loan_id": [3, 4],
            "portfolio_code": [5, 6],
            "portfolio_date": [2022, 2022],
            "company_name": ["Company A", "Company B"],
            "parent_name": ["Parent A", "Parent B"],
            "company_country": ["US", "DE"],
            "company_lei": ["123", "456"],
            "parent_lei": ["789", "012"],
            "outstanding_amount": [10000000, 900000],
        }
    )
    result = preparer._merge_climate_loan_data(sample_climate_data, sample_loan_data)
    assert_frame_equal(result, expected)


def test_process_name(preparer):
    names = pd.Series(["Company A, Inc.", "The Corporation B"])
    expected = pd.Series(["co a inc", "the corp b"])
    result = preparer._process_name(names)
    assert all(result == expected)


def test_preprocess_climate(preparer, sample_climate_data):
    preparer._climate = sample_climate_data
    preparer._preprocess_climate(True)
    assert "processed_name" in preparer._climate.columns


def test_preprocessed_names(preparer, sample_loan_data):
    preparer._preprocessed_names(sample_loan_data)
    assert "processed_name" in sample_loan_data.columns
    assert "processed_parent_name" in sample_loan_data.columns


def test_join(preparer, sample_loan_data):
    preparer._climate = pd.DataFrame(
        {
            "company_id": [1, 2],
            "processed_name": ["company a", "company b"],
            "lei": ["123", "456"],
        }
    )
    sample_loan_data["counterparty_id"] = [3, 4]
    sample_loan_data["company_lei"] = ["123", "999"]
    sample_loan_data["parent_lei"] = ["888", "456"]
    sample_loan_data["processed_name"] = ["company a", "company b"]
    sample_loan_data["processed_parent_name"] = ["parent a", "company b"]
    result = preparer._join(sample_loan_data)
    assert len(result)==2


def test_postprocess_join(preparer):
    combined = pd.DataFrame(
        {
            "counterparty_id": [1, 1],
            "company_id": [1, 1],
            "company_name": ["Company A", "Company A"],
            "name_company": ["Company A", "Company A"],
            "company_country": ["Country A", "Country A"],
            "company_lei": ["123", "123"],
            "parent_name": ["Parent A", "Parent A"],
            "parent_lei": ["456", "456"],
        }
    )
    result = preparer._postprocess_join(combined)
    assert len(result) == 1


def test_simple_join(preparer, sample_climate_data):
    matching_ids = pd.DataFrame({"company_id": [3, 4], "counterparty_id": [1, 2]})
    preparer._climate = sample_climate_data
    result = preparer._simple_join(matching_ids)
    assert not result.empty


def test_post_processed(preparer, sample_loan_data, sample_climate_data):
    preparer._additional_columns = {}
    preparer._external_columns = {}
    sample_climate_data['counterparty_id'] = [1,2]
    loanbook = preparer._merge_climate_loan_data(sample_climate_data, sample_loan_data)

    result = preparer._post_processed(loanbook)
    pd.testing.assert_frame_equal(result.drop_duplicates(), result)
    assert all(
        result.columns
        == [
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
    )


def test_match_data(preparer, sample_loan_data):
    with patch.object(preparer, "_join", return_value=sample_loan_data):
        with patch.object(preparer, "_preprocessed_names", MagicMock()):
            result = preparer._match_data(sample_loan_data)
            assert_frame_equal(result, sample_loan_data)
