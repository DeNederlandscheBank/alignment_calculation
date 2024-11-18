import pandas as pd
import pytest
from unittest.mock import patch
from alignment_calculation.read_data import _load_loan_data, _load_loan_counterparties, _add_external_data

@pytest.fixture
def climate_data():
    return pd.DataFrame({
        'counterparty_id': [1, 2, 3],
        'some_other_data': [10, 20, 30]
    })

@pytest.fixture
def loan_data():
    return pd.DataFrame({
        'loan_id': ['1', '2', '3'],
        'counterparty_id': [1, 2, 4],
        'portfolio_date': ['202101', '202102', '202103'],
        'portfolio_code': ['A', 'B', 'C']
    })

@pytest.fixture
def columns_dict():
    return {'new_column': [100, 200, 300]}

def test__load_loan_data(climate_data):
    with patch('pandas.read_csv', return_value=pd.DataFrame({
        'loan_id': ['1', '2', '3'],
        'counterparty_id': [1, 2, 4],
        'portfolio_date': ['202101', '202102', '202103'],
        'portfolio_code': ['A', 'B', 'C']
    })) as mock_read_csv:
        result = _load_loan_data(climate_data, 2021, 1, ['A', 'B'], None, None)
        assert len(result) == 2
        assert result['loan_id'].iloc[0] == '1'
        mock_read_csv.assert_called_once()

def test__load_loan_counterparties():
    with patch('pandas.read_csv', return_value=pd.DataFrame({
        'counterparty_id': [1, 2, 3],
        'name': ['Company A', 'Company B', 'Company C']
    })) as mock_read_csv:
        result = _load_loan_counterparties()
        assert len(result) == 3
        assert result['name'].iloc[0] == 'Company A'
        mock_read_csv.assert_called_once()

def test__add_external_data(loan_data, columns_dict):
    result = _add_external_data(loan_data, columns_dict)
    assert 'new_column' in result.columns
    assert result['new_column'].iloc[0] == 100
    assert result['new_column'].iloc[1] == 200
    assert result['new_column'].iloc[2] == 300
