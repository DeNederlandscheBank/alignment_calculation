import pandas as pd
import pytest
from unittest.mock import Mock, patch, mock_open
from alignment_calculation.load_climate_data import (
    _harmonise_column_names,
    _load_scenario_data,
    _load_region_data,
    _load_loanbook_data,
    _preprocess_indicators,
    _load_main_climate_data
)

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'sector': ['Energy', 'Transport'],
        'technology': ['Coal', 'Electric'],
        'scenario_source': ['A', 'B']
    })

@pytest.fixture
def sample_scenario_data():
    return {
        "scenario_files": {
            "2020": {
                "base": {
                    "scenario_file_tms": "path/to/tms.csv",
                    "scenario_file_sda": "path/to/sda.csv"
                }
            }
        },
        "region_file": {
            "base": "path/to/region.csv"
        }
    }

@pytest.fixture
def sample_settings():
    return {
        "sectoral_approach": {
            "Energy": {"active": True},
            "Transport": {"active": False}
        }
    }

def test_harmonise_column_names(sample_dataframe):
    result = _harmonise_column_names(sample_dataframe)
    assert list(result.columns) == ['sector', 'technology', 'scenario_source']

@patch('pandas.read_csv')
def test_load_scenario_data(mock_read_csv, sample_scenario_data):
    mock_read_csv.return_value = pd.DataFrame({
        'scenario': ['base'],
        'scenario_source': ['A'],
        'technology': ['coal'],
        'emission_factor_unit': ['tCO2']
    })
    result = _load_scenario_data(sample_scenario_data)
    assert isinstance(result, dict)
    assert '2020' in result
    assert 'base' in result['2020']

@patch('pandas.read_csv')
def test_load_region_data(mock_read_csv, sample_scenario_data):
    mock_read_csv.return_value = pd.DataFrame({
        'Region': ['EU', 'US'],
        'Classification': ['Developed', 'Developed']
    })
    result = _load_region_data(sample_scenario_data)
    assert isinstance(result, dict)
    assert 'base' in result

def test_load_loanbook_data_none():
    result = _load_loanbook_data(None)
    assert result is None

@patch('pandas.read_csv')
def test_load_loanbook_data(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        'Loan ID': [1, 2],
        'Amount': [1000, 2000]
    })
    result = _load_loanbook_data('path/to/loan.csv')
    assert isinstance(result, pd.DataFrame)
    assert 'loan_id' in result.columns

def test_preprocess_indicators(sample_dataframe, sample_settings):
    result = _preprocess_indicators(sample_dataframe, sample_settings)
    assert len(result) == 1
    assert result['sector'].iloc[0] == 'Energy'

class mock_load():
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, a=None, b=None, c=None):
        pass

@patch('pandas.ExcelFile')
@patch('pandas.read_excel')
@patch('pandas.read_csv')
def test_load_main_climate_data_excel(mock_read_csv, mock_read_excel, mock_ExcelFile, sample_settings):
    mock_read_excel.return_value = pd.DataFrame({
        'company_id': [1, 2],
        'sector': ['oil', 'gas'],
        'emissions': [100, 200]
    })
    mock_read_csv.return_value = pd.DataFrame({
        'company_id': [1, 2],
        'ownership': ['Private', 'Public']
    })
    mock_ExcelFile.return_value = mock_load()
    result = _load_main_climate_data('path/to/file.xlsx', settings=sample_settings)
    assert isinstance(result, dict)
    assert 'company_indicators' in result
    assert 'company_ownership' in result

@patch('pandas.read_csv')
def test_load_main_climate_data_csv(mock_read_csv, sample_settings):
    mock_read_csv.side_effect = [
        pd.DataFrame({
            'company_id': [1, 2],
            'sector': ['oil', 'gas'],
            'emissions': [100, 200]
        }),
        pd.DataFrame({
            'company_id': [1, 2],
            'ownership': ['Private', 'Public']
        })
    ]
    result = _load_main_climate_data(['path/to/indicators.csv', 'path/to/ownership.csv'], settings=sample_settings)
    assert isinstance(result, dict)
    assert 'company_indicators' in result
    assert 'company_ownership' in result