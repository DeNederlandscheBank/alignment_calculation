import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock
from alignment_calculation.calculator import alignmentCalculator

@pytest.fixture
def loan_indicator():
    return "outstanding_amount"

@pytest.fixture
def only_parents():
    return True

@pytest.fixture
def facet_col():
    return []

@pytest.fixture
def use_region_file():
    return True

@pytest.fixture
def normalise_method():
    return "total"

@pytest.fixture
def mock_preprocess_data():
    return pd.DataFrame({
        "company_id": [1, 2],
        "loan_indicator": [100, 200],
        "facet_col": ["sector1", "sector2"],
        "use_region_file": [True, True],
        "normalise_method": ["total", "total"]
    })

@pytest.fixture
def default_alignment_calculator():
    '''Fixture to create a default alignmentCalculator instance.'''
    return alignmentCalculator(portfolio_id="test_portfolio")

@pytest.fixture
def alignment_calculator():
    # Mocking the scenario data loading and other dependencies
    ac = alignmentCalculator(debug=True, portfolio_id="test_portfolio")
    ac._scenario_data = {
        2022: {
            'weo': {
                'nze_2050': {}
            }
        },
        2023: {
            'weo': {
                'nze_2050': {}
            }
        }
    }
    return ac

@pytest.fixture
def custom_alignment_calculator():
    '''Fixture to create a custom alignmentCalculator instance.'''
    custom_settings = {'custom_setting': 'value'}
    loan_file = pd.DataFrame({'column': [1, 2, 3]})
    loanbook_settings = {'base_year': 2022}
    return alignmentCalculator(
        portfolio_id='custom_portfolio',
        custom_settings=custom_settings,
        loan_file=loan_file,
        loanbook_settings=loanbook_settings,
        scenario_set='custom_scenario',
        pathway='custom_pathway',
        debug=True
    )


@pytest.fixture
def sample_data():
    # Sample data for testing
    data = {
        "company_id": ["C1", "C2", "C3"],
        "name_company": ["Company1", "Company2", "Company3"],
        "sector": ["Energy", "Energy", "Transport"],
        "technology": ["Coal", "Oil", "Electric"],
        "plant_location": ["USA", "Canada", "Mexico"],
        "year": [2023, 2023, 2023],
        "production": [100, 150, 200],
        "emission_factor": [0.1, 0.2, 0.3],
        "region": ["North America", "North America", "North America"],
        "target": [90, 140, 190],
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def setup_alignment_calculator():
    # Mock the alignmentCalculator with necessary attributes
    ac = alignmentCalculator(portfolio_id="test_portfolio")
    ac._loans = pd.DataFrame({
        'company_id': [1, 2],
        'loan_id': ['L1', 'L2'],
        'portfolio_date': [202301, 202301],
        'outstanding_amount': [100, 300]
    })
    ac._climate_company_indicators = {
        2023: pd.DataFrame({
            'company_id': [1, 2],
            'technology': ['tech1', 'tech2'],
            'sector': ['sector1', 'sector2'],
            'production': [100, 200],
            'target': [90, 180]
        })
    }
    ac._df_climate = {
        2023: pd.DataFrame({
            'company_id': [1, 2],
            'technology': ['tech1', 'tech2'],
            'sector': ['sector1', 'sector2'],
            'production': [100, 200],
            'target': [90, 180]
        })
    }
    ac._scenario_data = {
        2023: {
            'weo': {
                'nze_2050': pd.DataFrame({
                    'sector': ['sector1', 'sector2'],
                    'technology': ['tech1', 'tech2'],
                    'year': [2023, 2023],
                    'region': ['region1', 'region2'],
                    'target': [85, 175]
                })
            }
        }
    }
    return ac


def test_default_initialization(default_alignment_calculator):
    '''Test the default initialization of alignmentCalculator.'''
    assert default_alignment_calculator._portfolio_id == 'portfolio_code'
    assert default_alignment_calculator._scenario_set == 'weo'
    assert default_alignment_calculator._pathway == 'nze_2050'
    assert default_alignment_calculator._settings is not None
    assert not hasattr(default_alignment_calculator, '_loans'), "Loans should not be loaded in default initialization"

def test_custom_initialization(custom_alignment_calculator):
    '''Test the custom initialization of alignmentCalculator.'''
    assert custom_alignment_calculator._portfolio_id == 'custom_portfolio'
    assert custom_alignment_calculator._scenario_set == 'custom_scenario'
    assert custom_alignment_calculator._pathway == 'custom_pathway'
    assert custom_alignment_calculator._settings is not None
    assert isinstance(custom_alignment_calculator._loans, pd.DataFrame), "Loans should be loaded as DataFrame"

@patch('your_module._load_loanbook_data')
def test_loan_file_path_initialization(mock_load_loanbook_data):
    '''Test initialization with a loan file path.'''
    mock_load_loanbook_data.return_value = pd.DataFrame({'column': [1, 2, 3]})
    ac = alignmentCalculator(loan_file='path/to/loanfile.csv')
    mock_load_loanbook_data.assert_called_once_with('path/to/loanfile.csv')
    assert isinstance(ac._loans, pd.DataFrame), "Loans should be loaded from the specified path"

def test_debug_mode_initialization():
    '''Test initialization with debug mode enabled.'''
    ac = alignmentCalculator(debug=True)
    assert ac._scenario_data is None, "Debug mode should be enabled"
    assert len(ac._climate_ownership) == 0, "Debug mode should be enabled"
    assert len(ac._climate_company_indicators) == 0, "Debug mode should be enabled"
    assert not hasattr(ac, '_scenario_data'), "Scenario data should not be loaded in debug mode"

def test_calculate_net_alignment_default(alignment_calculator_instance):
    with patch.object(alignment_calculator_instance, '_preprocess_data', return_value=pd.DataFrame({
        'portfolio_date': [202301, 202302],
        'company_id': [1, 2],
        'sector': ['Energy', 'Transport'],
        'technology': ['Coal', 'Electric'],
        'year': [2023, 2023],
        'production': [100, 200],
        'target': [90, 180]
    })) as mock_preprocess, \
    patch.object(alignment_calculator_instance, '_calculate_alignment_instance', return_value=pd.DataFrame({
        'portfolio_date': [202301, 202302],
        'score': [0.1, 0.2]
    })) as mock_calculate_instance:
        
        result = alignment_calculator_instance.calculate_net_alignment()
        
        expected_result = pd.DataFrame({
            'portfolio_date': [202301, 202302],
            'score': [0.1, 0.2]
        })
        
        assert_frame_equal(result.data, expected_result)
        mock_preprocess.assert_called_once()
        mock_calculate_instance.assert_called()

def test_calculate_net_alignment_custom_params(alignment_calculator_instance):
    with patch.object(alignment_calculator_instance, '_preprocess_data', return_value=pd.DataFrame({
        'portfolio_date': [202301],
        'company_id': [1],
        'sector': ['Energy'],
        'technology': ['Coal'],
        'year': [2023],
        'production': [100],
        'target': [95]
    })) as mock_preprocess, \
    patch.object(alignment_calculator_instance, '_calculate_alignment_instance', return_value=pd.DataFrame({
        'portfolio_date': [202301],
        'score': [0.05]
    })) as mock_calculate_instance:
        
        result = alignment_calculator_instance.calculate_net_alignment(
            loan_indicator="total_assets",
            facet_col=["sector"],
            bopo_split=True,
            individual_loans=True,
            use_loan_file=False,
            only_parents=False,
            use_region_file=False,
            limit=5,
            normalise_method="economic"
        )
        
        expected_result = pd.DataFrame({
            'portfolio_date': [202301],
            'score': [0.05]
        })
        
        assert_frame_equal(result.data, expected_result)
        mock_preprocess.assert_called_once_with(
            use_loan_file=False,
            individual_loans=True,
            loan_indicator="total_assets",
            only_parents=False,
            facet_col=["sector"],
            use_region_file=False,
            year=2023,
            normalise_method="economic"
        )
        mock_calculate_instance.assert_called()

def test_calculate_net_alignment_change_over_time(setup_alignment_calculator):
    ac = setup_alignment_calculator
    result = ac.calculate_net_alignment_change_over_time()
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert not result.empty, "Result DataFrame should not be empty"
    assert 'score' in result.columns, "Result should have a 'score' column"
    assert result['score'].dtype == float, "Scores should be floats"
    assert all(result['score'] <= 3) and all(result['score'] >= -3), "Scores should be within the limit [-3, 3]"


def test_get_available_scenarios(alignment_calculator):
    # Expected DataFrame structure
    expected_df = pd.DataFrame({
        'scenario_set': ['weo', 'weo'],
        'pathway': ['nze_2050', 'nze_2050'],
        'year': [2022, 2023]
    })
    
    # Run the method
    result_df = alignment_calculator.get_available_scenarios()
    
    # Check if the result is as expected
    pd.testing.assert_frame_equal(result_df.sort_values(by='year').reset_index(drop=True), 
                                  expected_df.sort_values(by='year').reset_index(drop=True))

def test_set_scenarios(alignment_calculator_instance):
    # Mocking the scenario data loading
    alignment_calculator_instance._scenario_data = {
        2022: {
            'weo': {
                'nze_2050': MagicMock()
            }
        }
    }
    
    # Set scenarios
    scenario_set = 'weo'
    pathway = 'nze_2050'
    alignment_calculator_instance.set_scenarios(scenario_set, pathway)
    
    # Check if the scenario set and pathway are set correctly
    assert alignment_calculator_instance._scenario_set == scenario_set
    assert alignment_calculator_instance._pathway == pathway


def test_update_settings(alignment_calculator_instance):
    # Mock the alignmentCalculatorConfig class and its methods
    with pytest.MonkeyPatch.context() as m:
        mock_config = MagicMock()
        m.setattr("alignment_calculator.alignmentCalculatorConfig", MagicMock(return_value=mock_config))
        mock_config.load_settings.return_value = {'some_setting': 'original_value'}
        mock_config.config.return_value = None
        
        # Test updating settings
        new_settings = {'some_setting': 'new_value'}
        alignment_calculator_instance.update_settings(new_settings)
        
        # Assert that the config method was called with the correct parameters
        mock_config.config.assert_called_once_with(
            main_climate_file=None,
            company_information_file=None,
            economic_weights=None,
            production_thresholds=None,
            scenario_data=None,
            sectoral_approach=None,
            save_changes=True
        )
        
        # Assert that the settings were updated correctly
        assert alignment_calculator_instance._settings['some_setting'] == 'new_value'
        
        # Test that the settings are reloaded if config returns None
        assert mock_config.load_settings.called


@pytest.fixture
def loanbook_data():
    return pd.DataFrame({
        'company_id': [1, 2, 3],
        'loan_amount': [1000, 2000, 3000]
    })

@pytest.fixture
def loanbook_settings():
    return {
        'base_year': 2023,
        'month': 1,
        'loan_file': 'test_loans.csv',
        'additional_instrument_columns': {'column1': 'value1', 'column2': 'value2'}
    }

def test_update_loanbook_with_file(alignment_calculator_instance, loanbook_data):
    with patch('alignment_calculator._load_loanbook_data', return_value=loanbook_data) as mock_load:
        alignment_calculator_instance.update_loanbook(loan_file='test_loans.csv')
        mock_load.assert_called_once_with('test_loans.csv')
        assert isinstance(alignment_calculator_instance._loans, pd.DataFrame)
        assert not alignment_calculator_instance._loans.empty
        assert alignment_calculator_instance._loans.equals(loanbook_data)

def test_update_loanbook_with_settings(alignment_calculator_instance, loanbook_settings):
    with patch('alignment_calculator.loanbookPreparer') as mock_preparer:
        preparer_instance = mock_preparer.return_value
        preparer_instance.prepare_loanbook.return_value = pd.DataFrame({
            'company_id': [1, 2, 3],
            'loan_amount': [1000, 2000, 3000]
        })
        alignment_calculator_instance.update_loanbook(loanbook_settings=loanbook_settings)
        preparer_instance.prepare_loanbook.assert_called_once_with(**loanbook_settings)
        assert isinstance(alignment_calculator_instance._loans, pd.DataFrame)
        assert not alignment_calculator_instance._loans.empty


def test_preprocess_data(alignment_calculator_instance, mock_loan_data, mock_climate_data):
    with patch.object(alignment_calculator_instance, '_load_loanbook_data', return_value=mock_loan_data), \
         patch.dict(alignment_calculator_instance._climate_company_indicators, mock_climate_data['company_indicators']), \
         patch.dict(alignment_calculator_instance._scenario_data, mock_climate_data['scenario_data']):
        
        result = alignment_calculator_instance._preprocess_data(
            use_loan_file=True,
            individual_loans=False,
            loan_indicator='loan_amount',
            only_parents=True,
            facet_col=['sector'],
            use_region_file=False,
            year=2023,
            normalise_method='total'
        )
        
        assert not result.empty, "The result should not be empty."
        assert 'target' in result.columns, "The result should have a 'target' column."
        assert result['target'].equals(mock_climate_data['scenario_data'][2023]['weo']['nze_2050']['target']), \
            "The targets should match the scenario data."


def test_calculate_net_alignment_2(setup_alignment_calculator, mock_climate_data, mock_loan_data):
    '''
    Test the `calculate_net_alignment` method of the alignmentCalculator class.
    '''
    calculator = setup_alignment_calculator
    
    # Mocking the loading of climate and loan data
    with patch.object(calculator, '_climate_company_indicators', return_value={2023: mock_climate_data}), \
         patch.object(calculator, '_loans', return_value=mock_loan_data):
        
        # Run the calculate_net_alignment method
        result = calculator.calculate_net_alignment()
        
        # Check if the result is as expected
        assert isinstance(result, pd.DataFrame), "The result should be a pandas DataFrame."
        assert not result.empty, "The result DataFrame should not be empty."
        assert 'score' in result.columns, "The result DataFrame should contain a 'score' column."
        assert result['score'].dtype == float, "The scores should be floating point numbers."

        # Additional checks can be made based on expected values
        # For example:
        # assert result['score'].iloc[0] == expected_value, "The score for the first entry is not as expected."

def test_add_region(setup_alignment_calculator):
    # Setup
    calculator = setup_alignment_calculator
    year = 2023
    region_mapping = {
        'sector1': {'region1': ['Country1', 'Country2']},
        'sector2': {'region2': ['Country3']}
    }
    
    # Mock the _reconcile_regions method to return our predefined region_mapping
    with patch.object(calculator, '_reconcile_regions', return_value=region_mapping):
        # Execute the method under test
        calculator._add_region(year)
        
        # Assertions to check if regions are added correctly
        assert 'region' in calculator._climate_company_indicators[year].columns, "Region column should be added"
        assert calculator._climate_company_indicators[year].loc[
            calculator._climate_company_indicators[year]['sector'] == 'sector1', 'region'
        ].unique()[0] == 'region1', "Region should be 'region1' for sector1"
        assert calculator._climate_company_indicators[year].loc[
            calculator._climate_company_indicators[year]['sector'] == 'sector2', 'region'
        ].unique()[0] == 'region2', "Region should be 'region2' for sector2"


def test_reconcile_regions_no_region_data(setup_alignment_calculator):
    '''Test _reconcile_regions method when no region data is loaded.'''
    setup_alignment_calculator._regions = None
    result = setup_alignment_calculator._reconcile_regions()
    assert result is None, "Expected None when no region data is available"

def test_reconcile_regions_with_region_data(setup_alignment_calculator):
    '''Test _reconcile_regions method with mock region data.'''
    # Mocking region data
    setup_alignment_calculator._regions = {
        'weo': pd.DataFrame({
            'region': ['global', 'EU', 'NA'],
            'isos': ['ALL', 'BE,DE,FR', 'US,CA']
        })
    }
    setup_alignment_calculator._scenario_set = 'weo'
    setup_alignment_calculator._scenario_data = {
        2023: {
            'weo': {
                'nze_2050': pd.DataFrame({
                    'sector': ['energy', 'energy', 'transport'],
                    'region': ['global', 'EU', 'NA'],
                    'year': [2023, 2023, 2023]
                })
            }
        }
    }
    setup_alignment_calculator._settings = {
        'sectoral_approach': {
            'energy': {'regional': True},
            'transport': {'regional': False}
        }
    }

    expected = {
        'energy': {'global': ['ALL'], 'EU': ['BE', 'DE', 'FR'], 'NA': ['US', 'CA']},
        'transport': {'global': ['ALL']}
    }
    
    result = setup_alignment_calculator._reconcile_regions()
    assert result == expected, "Expected region mapping did not match the actual result"


def test_calculate_tms(setup_alignment_calculator):
    ac = setup_alignment_calculator
    
    # Mocking the necessary data and methods for the test
    with patch.object(ac, '_climate_company_indicators', return_value={2023: pd.DataFrame({
        'company_id': [1, 2],
        'name_company': ['Company A', 'Company B'],
        'sector': ['Energy', 'Transport'],
        'technology': ['Coal', 'Electric'],
        'plant_location': ['USA', 'Germany'],
        'year': [2023, 2023],
        'production': [1000, 2000],
        'emission_factor': [0.5, 0.3],
        'region': ['NA', 'EU']
    })}), patch.object(ac, '_scenario_data', return_value={
        2023: {
            'weo': {
                'nze_2050': pd.DataFrame({
                    'sector': ['Energy', 'Transport'],
                    'technology': ['Coal', 'Electric'],
                    'year': [2023, 2023],
                    'region': ['NA', 'EU'],
                    'target': [900, 1800]
                })
            }
        }
    }), patch.object(ac, '_settings', return_value={
        'sectoral_approach': {
            'Energy': {
                'approach': 'tms',
                'sector': ['Coal'],
                'technology': ['Coal'],
                'build_out': [],
                'phase_out': ['Coal'],
                'other': [],
                'regional': True,
                'active': True
            },
            'Transport': {
                'approach': 'sda',
                'sector': ['Electric'],
                'technology': ['Electric'],
                'build_out': [],
                'phase_out': [],
                'other': ['Electric'],
                'regional': True,
                'active': True
            }
        }
    }):
        # Run the method under test
        result = ac._calculate_tms(2023, 'weo', 'nze_2050')
        
        # Check the results
        assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."
        assert 'target' in result.columns, "The result DataFrame should contain 'target' column."
        assert result.at[0, 'target'] == 900, "The target for Energy sector should be calculated correctly."
        assert result.at[1, 'target'] == 1800, "The target for Transport sector should be calculated correctly."


def test_calculate_sda(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_climate = ac._climate_company_indicators[2023]
    sector = 'energy'
    scenario_year = 2023
    
    # Run the method under test
    result_df = ac._calculate_sda(df_climate, sector, scenario_year)
    
    # Define expected DataFrame
    expected_df = pd.DataFrame({
        'company_id': [1, 2],
        'sector': ['energy', 'transport'],
        'technology': ['solar', 'electric'],
        'production': [100, 200],
        'emission_factor': [0.1, 0.2],
        'region': ['EU', 'NA'],
        'target': [0.05, pd.NA]  # Only 'energy' sector should have updated targets
    })
    
    # Check if the result is as expected
    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.fixture
def climate_data():
    return pd.DataFrame({
        'company_id': [1, 2, 3],
        'name_company': ['Company A', 'Company B', 'Company C'],
        'sector': ['Energy', 'Energy', 'Technology'],
        'technology': ['Coal', 'Solar', 'AI'],
        'year': [2020, 2020, 2020],
        'region': ['NA', 'EU', 'APAC'],
        'production': [100, 200, 300],
        'target': [90, 180, 270]
    })


def test_combine_asset_locations(climate_data, alignment_calculator_instance):
    alignment_calculator_instance._df_climate = {2020: climate_data}
    expected_result = pd.DataFrame({
        'company_id': [1, 2, 3],
        'name_company': ['Company A', 'Company B', 'Company C'],
        'sector': ['Energy', 'Energy', 'Technology'],
        'technology': ['Energy', 'Energy', 'Technology'],
        'year': [2020, 2020, 2020],
        'region': ['NA', 'EU', 'APAC'],
        'production': [100, 200, 300],
        'target': [90, 180, 270]
    })

    result = alignment_calculator_instance._combine_asset_locations(2020)
    assert_frame_equal(result, expected_result)

def test_get_sector_approach_technologies(setup_alignment_calculator):
    ac = setup_alignment_calculator
    expected_technologies = ["technology_a", "technology_b", "technology_c"]
    result_technologies = ac._get_sector_approach_technologies()
    assert set(result_technologies) == set(expected_technologies), "The technologies should match the expected list"

@pytest.fixture
def mock_climate_data():
    return pd.DataFrame({
        'company_id': [1, 2, 3],
        'production': [100, 200, 300],
        'target': [90, 190, 290]
    })

@pytest.fixture
def mock_loan_data():
    return pd.DataFrame({
        'company_id': [1, 2, 3],
        'loan_amount': [1000, 2000, 3000]
    })

@patch('alignment_calculator_module.alignmentCalculator._load_region_data')
@patch('alignment_calculator_module.alignmentCalculator._preprocess_data')
def test_combine_climate_loan_data(mock_preprocess_data, mock_load_region_data, mock_alignment_calculator, mock_climate_data, mock_loan_data):
    mock_preprocess_data.return_value = mock_climate_data
    mock_load_region_data.return_value = pd.DataFrame({
        'region': ['Region1', 'Region2', 'Region3'],
        'company_id': [1, 2, 3]
    })
    
    result = mock_alignment_calculator._combine_climate_loan_data(
        climate_data=mock_climate_data,
        use_loan_file=True,
        individual_loans=False,
        only_parents=True,
        loan_column='loan_amount',
        facet_col=[],
        year=2023
    )
    
    expected_columns = ['company_id', 'production', 'target', 'loan_amount', 'portfolio_id', 'loan_id']
    assert all(col in result.columns for col in expected_columns), "Result should have all expected columns"
    assert len(result) == 3, "Result should have combined data for 3 companies"
    assert result['loan_amount'].equals(mock_loan_data['loan_amount']), "Loan amounts should match the input loan data"

def test_only_parents_true(setup_alignment_calculator):
    ac = setup_alignment_calculator
    result = ac._only_parents(ac._loans, 'outstanding_amount', 2023, only_parents=True)
    expected_result = pd.DataFrame({
        'company_id': ['P1', 'P2'],
        'loan_id': [1, 2],
        'outstanding_amount': [400, 600],  # Sum of loans for each parent
        'portfolio_date': [202301, 202301],
        'sector': ['Energy', 'Transport']
    })
    pd.testing.assert_frame_equal(result, expected_result)

def test_only_parents_false(setup_alignment_calculator):
    ac = setup_alignment_calculator
    result = ac._only_parents(ac._loans, 'outstanding_amount', 2023, only_parents=False)
    expected_result = pd.DataFrame({
        'company_id': ['C1', 'C2', 'C3', 'C4'],
        'loan_id': [1, 2, 3, 4],
        'outstanding_amount': [100, 200, 300, 400],
        'portfolio_date': [202301, 202301, 202301, 202301],
        'sector': ['Energy', 'Transport', 'Energy', 'Transport']
    })
    pd.testing.assert_frame_equal(result, expected_result)

def test_get_parent_companies(setup_alignment_calculator):
    '''Test the _get_parent_companies method of alignmentCalculator.'''
    ac = setup_alignment_calculator
    
    # Mocking the _climate_ownership data for the year 2023
    ac._climate_ownership = {
        2023: pd.DataFrame({
            'company_id': ['C1', 'C2', 'C3'],
            'parent_company_id': ['P1', 'P2', 'P3'],
            'is_parent': [True, True, True],
            'is_ultimate_listed_parent': [False, True, False],
            'is_ultimate_parent': [True, False, True],
            'ownership_level': [1, 1, 1]
        })
    }
    
    # Expected DataFrame
    expected_df = pd.DataFrame({
        'parent_company_id': ['P1', 'P2', 'P3']
    }, index=pd.Index(['C1', 'C2', 'C3'], name='company_id'))
    
    # Run the method
    result_df = ac._get_parent_companies(stop_at_weak_parents=True, year=2023)
    
    # Check if the result is as expected
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_apply_production_thresholds(setup_alignment_calculator):
    ac = setup_alignment_calculator
    
    # Create a sample DataFrame that mimics the expected structure
    data = {
        'company_id': ['C1', 'C2', 'C3', 'C4'],
        'sector': ['sector1', 'sector1', 'sector2', 'sector2'],
        'production': [100, 200, 300, 400],
        'total_assets': [50, 250, 350, 450],
        'turnover': [150, 250, 350, 450]
    }
    loan_data = pd.DataFrame(data)
    
    # Set production thresholds in the settings
    ac._settings['production_thresholds'] = {
        'sector1': {'asset_ratio': 0.5, 'turnover_ratio': 0.5},
        'sector2': {'asset_ratio': 0.8, 'turnover_ratio': 0.7}
    }
    
    # Expected DataFrame after applying production thresholds
    expected_data = {
        'company_id': ['C2', 'C3', 'C4'],
        'sector': ['sector1', 'sector2', 'sector2'],
        'production': [200, 300, 400],
        'total_assets': [250, 350, 450],
        'turnover': [250, 350, 450]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Apply production thresholds
    result_df = ac._apply_production_thresholds(loan_data)
    
    # Assert that the resulting DataFrame is as expected
    assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


@pytest.mark.parametrize("test_input,expected", [
    (
        pd.DataFrame({
            'company_id': ['comp1', 'comp2', 'comp3'],
            'sector': ['sector1', 'sector1', 'sector2'],
            'total_assets': [100, 200, 300],
            'turnover': [150, 250, 350],
            'production': [80, 120, 90]
        }),
        pd.DataFrame({
            'company_id': ['comp1', 'comp2', 'comp3'],
            'sector': ['sector1', 'sector1', 'sector2'],
            'total_assets': [100, 200, 300],
            'turnover': [150, 250, 350],
            'production': [80, 120, 90],
            'asset_ratio': [0.8, 0.6, 0.3],
            'turnover_ratio': [0.5333, 0.48, 0.2571]
        })
    ),
    (
        pd.DataFrame({
            'company_id': ['comp1', 'comp2'],
            'sector': ['sector2', 'sector2'],
            'total_assets': [100, 200],
            'turnover': [150, 250],
            'production': [10, 20]
        }),
        pd.DataFrame({
            'company_id': ['comp1', 'comp2'],
            'sector': ['sector2', 'sector2'],
            'total_assets': [100, 200],
            'turnover': [150, 250],
            'production': [10, 20],
            'asset_ratio': [0.1, 0.1],
            'turnover_ratio': [0.0667, 0.08]
        })
    )
])
def test_determine_ratios(setup_alignment_calculator, test_input, expected):
    ac = setup_alignment_calculator
    result = ac._determine_ratios(test_input)
    pd.testing.assert_frame_equal(result.round(4), expected.round(4))


@pytest.fixture
def sample_loan_data():
    data = {
        'company_id': [1, 2, 1, 2],
        'sector': ['Energy', 'Energy', 'Transport', 'Transport'],
        'production': [100, 200, 300, 400],
        'loan_amount': [1000, 2000, 1500, 2500]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_climate_company_indicators():
    data = {
        'company_id': [1, 2, 1, 2],
        'sector': ['Energy', 'Energy', 'Transport', 'Transport'],
        'production': [120, 180, 320, 380]
    }
    return pd.DataFrame(data)


def test_split_loans_over_sector(alignment_calculator_instance, sample_loan_data, sample_climate_company_indicators):
    alignment_calculator_instance._climate_company_indicators = {2023: sample_climate_company_indicators}
    
    expected_data = {
        'company_id': [1, 2, 1, 2],
        'sector': ['Energy', 'Energy', 'Transport', 'Transport'],
        'production': [100, 200, 300, 400],
        'loan_amount': [1000 * 120/(120+180), 2000 * 180/(120+180), 1500 * 320/(320+380), 2500 * 380/(320+380)]
    }
    expected_df = pd.DataFrame(expected_data)
    
    result_df = alignment_calculator_instance._split_loans_over_sector(sample_loan_data, 'loan_amount', 2023)
    
    assert_frame_equal(result_df, expected_df, check_dtype=False)


@pytest.fixture
def sample_climate_data():
    # Sample data simulating the climate data structure
    data = {
        "company_id": ["C1", "C2", "C3"],
        "sector": ["Energy", "Energy", "Transport"],
        "technology": ["Coal", "Solar", "Electric"],
        "production": [1000, 2000, 1500],
        "target": [900, 1900, 1400],
        "loan_indicator": [100, 200, 150]
    }
    return pd.DataFrame(data)

def test_split_over_technology(setup_alignment_calculator, sample_climate_data):
    # Get the alignmentCalculator instance from the fixture
    calculator = setup_alignment_calculator
    
    # Mocking the internal method calls that _split_over_technology might depend on
    calculator._get_sector_approach_technologies = MagicMock(return_value=["Coal", "Solar", "Electric"])
    
    # Call the method under test
    result = calculator._split_over_technology(sample_climate_data, "loan_indicator")
    
    # Check if the method outputs the expected DataFrame structure
    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."
    
    # Check if the columns are as expected
    expected_columns = set(["company_id", "sector", "technology", "production", "target", "loan_indicator"])
    assert set(result.columns) == expected_columns, "DataFrame should have specific columns."
    
    # Check if the technology split logic is correctly applied
    # This is a basic check assuming the function modifies 'loan_indicator' based on 'technology'
    assert all(result["loan_indicator"] != sample_climate_data["loan_indicator"]), "loan_indicator values should be modified based on technology."


def test_normalise_production_global(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_input = pd.DataFrame({
        'company_id': [1, 2],
        'production': [100, 200],
        'sector': ['sector1', 'sector2']
    })
    expected_output = pd.DataFrame({
        'company_id': [1, 2],
        'production': [100, 200],
        'sector': ['sector1', 'sector2'],
        'norm': [300, 300]  # Global normalisation based on total production
    })
    result = ac._normalise_production('global', df_input, 2020, ac._settings['economic_weights'], 'production')
    assert_frame_equal(result, expected_output)

def test_normalise_production_economic(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_input = pd.DataFrame({
        'company_id': [1, 2],
        'production': [100, 200],
        'sector': ['sector1', 'sector2']
    })
    expected_output = pd.DataFrame({
        'company_id': [1, 2],
        'production': [100, 200],
        'sector': ['sector1', 'sector2'],
        'norm': [100, 300]  # Economic weights applied
    })
    result = ac._normalise_production('economic', df_input, 2020, ac._settings['economic_weights'], 'production')
    assert_frame_equal(result, expected_output)


def test_global_normalisation(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    # Assume _global_normalisation is a method of alignmentCalculator
    # This is a hypothetical test; the actual implementation details might differ
    normalised_data = ac._global_normalisation(sample_data, 2023)
    
    # Check if the normalisation column exists
    assert 'norm' in normalised_data.columns, "Normalization column 'norm' should exist"
    
    # Check if the normalisation is done correctly
    # Here we assume the normalisation process sums up the 'production' and divides each by the sum
    total_production = sample_data['production'].sum()
    expected_norm = total_production / sample_data['production']
    pd.testing.assert_series_equal(normalised_data['norm'], expected_norm, check_names=False)


@pytest.fixture
def economic_weights():
    return {
        'sector1': 1.5,
        'sector2': 0.5
    }

@pytest.fixture
def df_combined():
    # Mock data simulating combined DataFrame after processing
    return pd.DataFrame({
        'company_id': ['C1', 'C2', 'C1', 'C2'],
        'sector': ['sector1', 'sector1', 'sector2', 'sector2'],
        'production': [100, 200, 300, 400],
        'year': [2023, 2023, 2023, 2023]
    })

def test_economic_normalisation(setup_calculator, df_combined, economic_weights):
    # Expected DataFrame after applying economic normalisation
    expected_df = pd.DataFrame({
        'company_id': ['C1', 'C2', 'C1', 'C2'],
        'sector': ['sector1', 'sector1', 'sector2', 'sector2'],
        'production': [100, 200, 300, 400],
        'year': [2023, 2023, 2023, 2023],
        'norm': [150.0, 150.0, 200.0, 200.0]  # Normalised by economic weights
    })

    # Run the economic normalisation method
    result_df = setup_calculator._economic_normalisation(df_combined, economic_weights, 2023)

    # Assert that the resulting DataFrame is as expected
    assert_frame_equal(result_df, expected_df)


@pytest.fixture
def setup_calculator():
    ac = alignmentCalculator()
    return ac

def test_total_normalisation(setup_calculator):
    ac = setup_calculator
    # Create a sample DataFrame to simulate combined data
    data = pd.DataFrame({
        'year': [2023, 2023, 2023],
        'sector': ['Energy', 'Energy', 'Transport'],
        'production': [100, 200, 300],
        'loan_indicator': [1000, 2000, 1500]
    })
    loan_indicator = 'loan_indicator'
    
    # Expected normalization should sum production for each sector and use it as norm
    expected_norms = pd.Series([300, 300, 300], name='norm')
    
    # Run the total_normalisation method
    normalized_data = ac._total_normalisation(data, loan_indicator, 2023)
    
    # Check if the 'norm' column matches the expected values
    pd.testing.assert_series_equal(normalized_data['norm'], expected_norms)


def test_portfolio_normalisation(setup_alignment_calculator):
    ac = setup_alignment_calculator
    # Create a sample DataFrame to simulate combined data
    data = pd.DataFrame({
        'year': [2023, 2023, 2023, 2023],
        'sector': ['Energy', 'Energy', 'Transport', 'Transport'],
        'company_id': [1, 2, 3, 4],
        'production': [100, 200, 300, 400],
        'loan_indicator': [1000, 2000, 3000, 4000],
        'portfolio_id': ['A', 'A', 'B', 'B']
    })
    
    # Expected DataFrame after normalisation
    expected_data = pd.DataFrame({
        'year': [2023, 2023, 2023, 2023],
        'sector': ['Energy', 'Energy', 'Transport', 'Transport'],
        'company_id': [1, 2, 3, 4],
        'production': [100, 200, 300, 400],
        'loan_indicator': [1000, 2000, 3000, 4000],
        'portfolio_id': ['A', 'A', 'B', 'B'],
        'norm': [300, 300, 700, 700]  # Normalisation factor based on sector and portfolio_id
    })
    
    # Run the normalisation method
    result_data = ac._portfolio_normalisation(data, 'loan_indicator', 2023)
    
    # Check if the resulting DataFrame is as expected
    assert_frame_equal(result_data, expected_data)



def test_company_normalisation(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    # Assume _company_normalisation is a method of alignmentCalculator
    # and it has been implemented to normalize 'production' based on 'loan_amount'
    
    # Expected DataFrame after normalization
    expected_data = {
        "company_id": ["C1", "C2", "C3"],
        "sector": ["Energy", "Energy", "Transport"],
        "production": [100 / 1000, 150 / 2000, 200 / 3000],
        "target": [90, 160, 190],
        "loan_amount": [1000, 2000, 3000],
        "norm": [1000, 2000, 3000]  # Assuming normalization factor is added as a new column
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Run the normalization method
    result_df = ac._company_normalisation(sample_data)
    
    # Check if the resulting DataFrame is as expected
    assert_frame_equal(result_df, expected_df)



def test_calculate_alignment_instance(setup_alignment_calculator):
    ac = setup_alignment_calculator
    data = pd.DataFrame({
        'company_id': [1, 2],
        'technology': ['tech1', 'tech2'],
        'sector': ['sector1', 'sector2'],
        'production': [100, 200],
        'target': [90, 180],
        'portfolio_date': [202312, 202312]
    })
    facet_col = []
    loan_indicator = 'outstanding_amount'
    bopo_split = False
    limit = 3
    
    # Call the method under test
    result = ac._calculate_alignment_instance(data, facet_col, loan_indicator, bopo_split, limit)
    
    # Check the results
    expected = pd.DataFrame({
        'end_year': [2023, 2023],
        'portfolio_date': [202312, 202312],
        'score': [0.111111, 0.111111]
    })
    
    pd.testing.assert_frame_equal(result, expected)


@pytest.fixture
def alignment_calculator_instance():
    ac = alignmentCalculator()
    ac._climate_company_indicators = {
        2023: pd.DataFrame({
            'company_id': [1, 2],
            'name_company': ['Company A', 'Company B'],
            'sector': ['Energy', 'Transport'],
            'technology': ['Coal', 'Electric'],
            'plant_location': ['USA', 'Germany'],
            'year': [2023, 2023],
            'production': [1000, 2000],
            'emission_factor': [0.5, 0.3],
            'region': ['NA', 'EU']
        })
    }
    ac._df_climate = {
        2023: pd.DataFrame({
            'company_id': [1, 2],
            'name_company': ['Company A', 'Company B'],
            'sector': ['Energy', 'Transport'],
            'technology': ['Coal', 'Electric'],
            'plant_location': ['USA', 'Germany'],
            'year': [2023, 2023],
            'production': [1000, 2000],
            'emission_factor': [0.5, 0.3],
            'region': ['NA', 'EU'],
            'target': [900, 1800]
        })
    }
    ac._scenario_data = {
        2023: {
            'weo': {
                'nze_2050': pd.DataFrame({
                    'sector': ['Energy', 'Transport'],
                    'technology': ['Coal', 'Electric'],
                    'year': [2023, 2023],
                    'region': ['NA', 'EU'],
                    'smsp': [0.9, 0.9]
                })
            }
        }
    }
    ac._settings = {
        'scenario_data': 'path/to/scenario_data',
        'main_climate_file': {
            2023: 'path/to/climate_file_2023'
        }
    }
    ac._portfolio_id = 'portfolio_code'
    return ac

def test_make_weighted_target(alignment_calculator_instance):
    ac = alignment_calculator_instance
    data = pd.DataFrame({
        'company_id': [1, 2],
        'year': [2023, 2023],
        'production': [1000, 2000],
        'target': [900, 1800],
        'loan_indicator': [100, 200]
    })
    cols = ['company_id', 'year']
    end_year = 2025
    result = ac._make_weighted_target(data, 'loan_indicator', cols, end_year)
    expected = pd.DataFrame({
        'company_id': [1, 2],
        'year': [2023, 2023],
        'production': [1000, 2000],
        'target': [900, 1800],
        'loan_indicator': [100, 200],
        'deviation': [100, 200],
        'weighted_deviation': [10000, 40000],
        'target_end': [900, 1800],
        'weighted_target': [90000, 360000]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_make_bopo_split(alignment_calculator_instance, sample_data):
    # Mocking the settings to include necessary sectoral approach details
    alignment_calculator_instance._settings = {
        'sectoral_approach': {
            'Energy': {
                'build_out': ['Coal'],
                'phase_out': ['Solar'],
                'other': []
            },
            'Transport': {
                'build_out': ['Electric'],
                'phase_out': ['Diesel'],
                'other': []
            }
        }
    }
    
    # Expected output
    expected_directions = ['build_out', 'phase_out', 'build_out', 'phase_out']
    
    # Running the method
    result = alignment_calculator_instance._make_bopo_split(sample_data)
    
    # Checking if the directions are correctly assigned
    assert list(result['direction']) == expected_directions, "The BoPo split directions are not assigned correctly."


def test_aggregate_results(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    ac._df_climate = {2023: sample_data}  # Mocking the climate data for 2023
    
    # Expected DataFrame after aggregation
    expected_data = {
        "end_year": [2023, 2023, 2023],
        "portfolio_date": [202312, 202312, 202312],
        "sector": ["Energy", "Energy", "Transport"],
        "technology": ["Coal", "Oil", "Electric"],
        "region": ["North America", "North America", "North America"],
        "score": [-0.111, -0.071, -0.053],
        "portfolio_id": ["test_portfolio", "test_portfolio", "test_portfolio"]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Running the actual aggregation method
    results = ac._calculate_alignment_instance(
        data=sample_data,
        facet_col=["sector", "technology", "region"],
        loan_indicator="production",
        bopo_split=False,
        limit=3,
        horzion=0
    )
    
    # Asserting if the results are as expected
    assert_frame_equal(results, expected_df, check_dtype=False)

def test_make_portfolio_dates(setup_alignment_calculator):
    ac = setup_alignment_calculator
    expected_dates = {202312, 202412}
    assert ac._make_portfolio_dates() == expected_dates, "The portfolio dates should match the expected dates ending with '12'"

@patch('alignment_calculator.alignmentCalculator._preprocess_data')
def test_make_master_data(mock_preprocess_data, alignment_calculator_instance, loan_indicator, only_parents, facet_col, use_region_file, normalise_method):
    mock_preprocess_data.return_value = pd.DataFrame({
        "company_id": [1, 2],
        "loan_indicator": [100, 200],
        "facet_col": ["sector1", "sector2"],
        "use_region_file": [True, True],
        "normalise_method": ["total", "total"]
    })
    
    result = alignment_calculator_instance._make_master_data(
        loan_indicator=loan_indicator,
        only_parents=only_parents,
        facet_col=facet_col,
        use_region_file=use_region_file,
        normalise_method=normalise_method
    )
    
    assert isinstance(result, dict)
    assert all(isinstance(v, pd.DataFrame) for v in result.values())
    assert all("company_id" in df.columns for df in result.values())
    assert all("loan_indicator" in df.columns for df in result.values())
    assert all("facet_col" in df.columns for df in result.values())
    assert all("use_region_file" in df.columns for df in result.values())
    assert all("normalise_method" in df.columns for df in result.values())
    mock_preprocess_data.assert_called()

def test_aggregate_over_time_results(alignment_calculator_instance, sample_data):
    with patch.object(alignment_calculator_instance, '_make_time_metrics', return_value=(sample_data, sample_data)):
        result = alignment_calculator_instance._aggregate_over_time_results(sample_data, 'loan_indicator', add_total=True)
        assert not result.empty, "The result should not be empty."
        assert 'total_shift' in result.columns, "The result should have a 'total_shift' column."
        assert result['total_shift'].notna().all(), "All values in 'total_shift' should be non-NaN."
        assert result['score'].notna().all(), "All values in 'score' should be non-NaN."
        assert (result['score'] <= 3).all() and (result['score'] >= -3).all(), "Scores should be clipped between -3 and 3."

def test_calculate_net_alignment(setup_alignment_calculator):
    # Setup
    calculator = setup_alignment_calculator
    
    # Define test data
    test_data = pd.DataFrame({
        'loan_indicator': ['outstanding_amount', 'outstanding_amount'],
        'facet_col': [['sector'], ['sector']],
        'bopo_split': [True, False],
        'individual_loans': [True, False],
        'use_loan_file': [True, False],
        'only_parents': [True, False],
        'use_region_file': [True, False],
        'limit': [3, 5],
        'normalise_method': ['total', 'economic']
    })
    
    # Expected results (mocked)
    expected_results = [
        pd.DataFrame({'result': [1, 2]}),
        pd.DataFrame({'result': [3, 4]})
    ]
    
    # Patch the method that would be called within calculate_net_alignment
    with patch.object(calculator, '_preprocess_data', side_effect=expected_results) as mock_method:
        # Run the test for each row in test_data
        for index, row in test_data.iterrows():
            result = calculator.calculate_net_alignment(
                loan_indicator=row['loan_indicator'],
                facet_col=row['facet_col'],
                bopo_split=row['bopo_split'],
                individual_loans=row['individual_loans'],
                use_loan_file=row['use_loan_file'],
                only_parents=row['only_parents'],
                use_region_file=row['use_region_file'],
                limit=row['limit'],
                normalise_method=row['normalise_method']
            )
            
            # Assert that the result matches the expected result
            pd.testing.assert_frame_equal(result, expected_results[index]) # type: ignore
            
            # Assert that _preprocess_data was called with the correct parameters
            mock_method.assert_called_with(
                use_loan_file=row['use_loan_file'],
                individual_loans=row['individual_loans'],
                loan_indicator=row['loan_indicator'],
                only_parents=row['only_parents'],
                facet_col=row['facet_col'],
                use_region_file=row['use_region_file'],
                year=2023,  # Assuming the default year is 2023
                normalise_method=row['normalise_method']
            )