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
def loanbook_data():
    return pd.DataFrame({"company_id": [1, 2, 3], "loan_amount": [1000, 2000, 3000]})


@pytest.fixture
def loanbook_settings():
    return {
        "base_year": 2023,
        "month": 1,
        "loan_file": "test_loans.csv",
        "additional_instrument_columns": {"column1": "value1", "column2": "value2"},
    }


@pytest.fixture
def mock_preprocess_data():
    return pd.DataFrame(
        {
            "company_id": [1, 2],
            "loan_indicator": [100, 200],
            "facet_col": ["sector1", "sector2"],
            "use_region_file": [True, True],
            "normalise_method": ["total", "total"],
        }
    )


@pytest.fixture
def default_alignment_calculator():
    """Fixture to create a default alignmentCalculator instance."""
    return alignmentCalculator(portfolio_id="portfolio_code")


@pytest.fixture
def alignment_calculator():
    # Mocking the scenario data loading and other dependencies
    ac = alignmentCalculator(debug=True, portfolio_id="portfolio_code")
    ac._scenario_data = {
        2022: {"weo": {"nze_2050": {}}},
        2023: {"weo": {"nze_2050": {}}},
    }
    return ac


@pytest.fixture
def custom_alignment_calculator():
    """Fixture to create a custom alignmentCalculator instance."""
    custom_settings = {
        "company_information_file": "../data/company_data/company_information.csv"
    }
    loan_file = pd.DataFrame({"column": [1, 2, 3]})
    loanbook_settings = {"base_year": 2022, "external_columns": None}
    return alignmentCalculator(
        portfolio_id="custom_portfolio",
        custom_settings=custom_settings,
        loan_file=loan_file,
        loanbook_settings=loanbook_settings,
        scenario_set="custom_scenario",
        pathway="custom_pathway",
        debug=False,
    )


@pytest.fixture
def sample_data():
    # Sample data for testing
    data = {
        "company_id": ["C1", "C2", "C3"],
        "name_company": ["Company1", "Company2", "Company3"],
        "sector": ["power", "power", "automotive"],
        "technology": ["coalcap", "oilcap", "electric"],
        "plant_location": ["US", "CA", "DE"],
        "year": [2023, 2023, 2023],
        "production": [100, 150, 200],
        "emission_factor": [0.1, 0.2, 0.3],
        "region": ["North America", "North America", "Europe"],
        "target": [90, 140, 190],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def combined_data():
    return None


@pytest.fixture(autouse=True)
def setup_alignment_calculator():
    # Mock the alignmentCalculator with necessary attributes
    ac = alignmentCalculator(portfolio_id="portfolio_code")
    ac._loans = pd.DataFrame(
        {
            "portfolio_code": ["portfolio1", "portfolio1"],
            "counterparty_id": ["C1", "C2"],
            "company_name": ["Company1", "Company2"],
            "name_company": ["Company1", "Company2"],
            "company_country": ["US", "IT"],
            "company_lei": ["123", "456"],
            "parent_name": ["Company1", "Parent1"],
            "parent_lei": ["123", "789"],
            "company_id": [88, 110],
            "loan_id": ["L1", "L2"],
            "portfolio_date": [202301, 202301],
            "outstanding_amount": [100, 300],
        }
    )
    ac._climate_company_indicators = {
        2023: pd.DataFrame(
            {
                "name_company": ["Company1", "Company2"],
                "lei": ["123", "456"],
                "plant_location": ["US", "DE"],
                "year": [2023, 2023],
                "production_unit": ["a", "b"],
                "emission_factor": [0.01, 0.02],
                "emission_factor_unit": ["a", "b"],
                "region": ["US", "EU"],
                "company_id": [88, 110],
                "sector": ["power", "automotive"],
                "technology": ["coalcap", "electric"],
                "production": [100, 200],
                "target": [90, 180],
            }
        )
    }
    ac._df_climate = {
        2023: pd.DataFrame(
            {
                "company_id": [88, 110],
                "name_company": ["Company1", "Company2"],
                "sector": ["power", "automotive"],
                "technology": ["coalcap", "electric"],
                "plant_location": ["US", "DE"],
                "year": [2023, 2023],
                "emission_factor": [0.01, 0.02],
                "smsp": [0.05, 0.05],
                "tmsr": [1.01, 1.23],
                "emission_factor_scenario": [1.01, 1.23],
                "region": ["US", "EU"],
                "production": [100, 200],
                "target": [90, 180],
            }
        )
    }
    ac._scenario_data = {
        2023: {
            "weo": {
                "nze_2050": pd.DataFrame(
                    {
                        "sector": ["power", "automotive"],
                        "technology": ["coalcap", "electric"],
                        "year": [2023, 2023],
                        "region": ["region1", "region2"],
                        "smsp": [0.05, 0.05],
                        "tmsr": [1.01, 1.23],
                        "emission_factor": [0.01, 0.89],
                        "target": [85, 175],
                    }
                )
            }
        }
    }
    return ac


def test_default_initialization(default_alignment_calculator):
    """Test the default initialization of alignmentCalculator."""
    assert default_alignment_calculator._portfolio_id == "portfolio_code"
    assert default_alignment_calculator._scenario_set == "weo"
    assert default_alignment_calculator._pathway == "nze_2050"
    assert default_alignment_calculator._settings is not None
    assert not hasattr(
        default_alignment_calculator, "_loans"
    ), "Loans should not be loaded in default initialization"


def test_custom_initialization(custom_alignment_calculator):
    """Test the custom initialization of alignmentCalculator."""
    assert custom_alignment_calculator._portfolio_id == "custom_portfolio"
    assert custom_alignment_calculator._scenario_set == "custom_scenario"
    assert custom_alignment_calculator._pathway == "custom_pathway"
    assert custom_alignment_calculator._settings is not None
    assert isinstance(custom_alignment_calculator._loans, pd.DataFrame)


@patch("alignment_calculation.calculator._load_loanbook_data")
def test_loan_file_path_initialization(mock_load_loanbook_data):
    """Test initialization with a loan file path."""
    mock_load_loanbook_data.return_value = pd.DataFrame({"column": [1, 2, 3]})
    ac = alignmentCalculator(loan_file="path/to/loanfile.csv")
    mock_load_loanbook_data.assert_called_once_with("path/to/loanfile.csv")
    assert isinstance(
        ac._loans, pd.DataFrame
    ), "Loans should be loaded from the specified path"


def test_debug_mode_initialization():
    """Test initialization with debug mode enabled."""
    ac = alignmentCalculator(debug=True)
    assert len(ac._climate_ownership) == 0, "Debug mode should be enabled"
    assert len(ac._climate_company_indicators) == 0, "Debug mode should be enabled"
    assert not hasattr(
        ac, "_scenario_data"
    ), "Scenario data should not be loaded in debug mode"


def test_calculate_net_alignment(setup_alignment_calculator):
    with (
        patch.object(
            setup_alignment_calculator,
            "_preprocess_data",
            return_value=pd.DataFrame(
                {
                    "portfolio_date": [202301, 202301],
                    "company_id": [1, 2],
                    "sector": ["Energy", "Transport"],
                    "technology": ["Coal", "Electric"],
                    "year": [2023, 2023],
                    "production": [100, 200],
                    "target": [90, 180],
                }
            ),
        ) as mock_preprocess,
        patch.object(
            setup_alignment_calculator,
            "_calculate_alignment_instance",
            return_value=pd.DataFrame(
                {"portfolio_date": [202301, 202301], "score": [0.1, 0.2]}
            ),
        ) as mock_calculate_instance,
    ):

        result = setup_alignment_calculator.calculate_net_alignment()

        expected_result = pd.DataFrame(
            {"portfolio_date": [202301, 202301], "score": [0.1, 0.2]}
        )

        assert_frame_equal(result._results_data, expected_result)
        mock_preprocess.assert_called_once()
        mock_calculate_instance.assert_called()


def test_calculate_net_alignment_change_over_time(setup_alignment_calculator):
    ac = setup_alignment_calculator
    result = ac.calculate_net_alignment_change_over_time()
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert not result.empty, "Result DataFrame should not be empty"
    assert "score" in result.columns, "Result should have a 'score' column"
    assert result["score"].dtype == float, "Scores should be floats"
    assert all(result["score"] <= 3) and all(
        result["score"] >= -3
    ), "Scores should be within the limit [-3, 3]"


def test_get_available_scenarios(alignment_calculator):
    # Expected DataFrame structure
    expected_df = pd.DataFrame(
        {
            "scenario_set": ["weo", "weo"],
            "pathway": ["nze_2050", "nze_2050"],
            "year": [2022, 2023],
        }
    )

    # Run the method
    result_df = alignment_calculator.get_available_scenarios()

    # Check if the result is as expected
    pd.testing.assert_frame_equal(
        result_df.sort_values(by="year").reset_index(drop=True),
        expected_df.sort_values(by="year").reset_index(drop=True),
    )


def test_set_scenarios(setup_alignment_calculator):
    # Mocking the scenario data loading
    setup_alignment_calculator._scenario_data = {
        2022: {"weo": {"nze_2050": MagicMock()}}
    }

    # Set scenarios
    scenario_set = "weo"
    pathway = "nze_2050"
    setup_alignment_calculator.set_scenarios(scenario_set, pathway)

    # Check if the scenario set and pathway are set correctly
    assert setup_alignment_calculator._scenario_set == scenario_set
    assert setup_alignment_calculator._pathway == pathway


def test_update_settings(setup_alignment_calculator):
    # Mock the alignmentCalculatorConfig class and its methods
    with pytest.MonkeyPatch.context() as m:
        mock_config = MagicMock()
        m.setattr(
            "alignment_calculation.calculator.alignmentCalculatorConfig",
            MagicMock(return_value=mock_config),
        )
        mock_config.load_settings.return_value = {
            "company_information_file": "original_value"
        }
        mock_config.config.return_value = None

        # Test updating settings
        new_settings = {"company_information_file": "new_value"}
        setup_alignment_calculator.update_settings(new_settings)

        # Assert that the config method was called with the correct parameters
        mock_config.config.assert_called_once_with(
            main_climate_file=None,
            company_information_file="new_value",
            economic_weights=None,
            production_thresholds=None,
            scenario_data=None,
            sectoral_approach=None,
            save_changes=True,
        )

        # Assert that the settings were updated correctly
        assert (
            setup_alignment_calculator._settings["company_information_file"]
            == "original_value"
        )

        # Test that the settings are reloaded if config returns None
        assert mock_config.load_settings.not_called


def test_update_loanbook_with_file(setup_alignment_calculator, loanbook_data):
    with patch(
        "alignment_calculation.calculator._load_loanbook_data",
        return_value=loanbook_data,
    ) as mock_load:
        setup_alignment_calculator.update_loanbook(loan_file="test_loans.csv")
        mock_load.assert_called_once_with("test_loans.csv")
        assert isinstance(setup_alignment_calculator._loans, pd.DataFrame)
        assert not setup_alignment_calculator._loans.empty
        assert setup_alignment_calculator._loans.equals(loanbook_data)


def test_update_loanbook_with_settings(setup_alignment_calculator, loanbook_settings):
    with patch("alignment_calculation.calculator.loanbookPreparer") as mock_preparer:
        preparer_instance = mock_preparer.return_value
        preparer_instance.prepare_loanbook.return_value = pd.DataFrame(
            {"company_id": [1, 2, 3], "loan_amount": [1000, 2000, 3000]}
        )
        setup_alignment_calculator.update_loanbook(loanbook_settings=loanbook_settings)
        preparer_instance.prepare_loanbook.assert_called_once_with(**loanbook_settings)
        assert isinstance(setup_alignment_calculator._loans, pd.DataFrame)
        assert not setup_alignment_calculator._loans.empty


def test_preprocess_data(setup_alignment_calculator, mock_loan_data, mock_climate_data):
    with (
        patch.object(
            setup_alignment_calculator,
            "_load_loanbook_data",
            return_value=mock_loan_data,
        ),
        patch.dict(
            setup_alignment_calculator._climate_company_indicators,
            mock_climate_data["company_indicators"],
        ),
        patch.dict(
            setup_alignment_calculator._scenario_data,
            mock_climate_data["scenario_data"],
        ),
    ):

        result = setup_alignment_calculator._preprocess_data(
            use_loan_file=True,
            individual_loans=False,
            loan_indicator="loan_amount",
            only_parents=True,
            facet_col=["sector"],
            use_region_file=False,
            year=2023,
            normalise_method="total",
        )

        assert not result.empty, "The result should not be empty."
        assert "target" in result.columns, "The result should have a 'target' column."
        assert result["target"].equals(
            mock_climate_data["scenario_data"][2023]["weo"]["nze_2050"]["target"]
        ), "The targets should match the scenario data."


def test_add_region(setup_alignment_calculator):
    # Setup
    calculator = setup_alignment_calculator
    year = 2023
    region_mapping = {
        "power": {"region1": ["US"]},
        "automotive": {"region2": ["US"]},
    }
    calculator._add_region(year, region_mapping)
    
    assert calculator._climate_company_indicators[2023]['region'].iloc[0] == 'region1'
    assert calculator._climate_company_indicators[2023]['region'].iloc[1] == 'global'

def test_reconcile_regions_no_region_data(setup_alignment_calculator):
    """Test _reconcile_regions method when no region data is loaded."""
    setup_alignment_calculator._regions = None
    result = setup_alignment_calculator._reconcile_regions()
    assert result is None, "Expected None when no region data is available"


def test_reconcile_regions_with_region_data(setup_alignment_calculator):
    """Test _reconcile_regions method with mock region data."""
    # Mocking region data
    setup_alignment_calculator._regions = {
        "weo": pd.DataFrame(
            {"region": ["global", "EU", "NA"], "isos": ["ALL", "BE,DE,FR", "US,CA"]}
        )
    }
    setup_alignment_calculator._scenario_set = "weo"
    setup_alignment_calculator._scenario_data = {
        2023: {
            "weo": {
                "nze_2050": pd.DataFrame(
                    {
                        "sector": ["energy", "energy", "transport"],
                        "region": ["global", "EU", "NA"],
                        "year": [2023, 2023, 2023],
                    }
                )
            }
        }
    }
    setup_alignment_calculator._settings = {
        "sectoral_approach": {
            "energy": {"regional": True},
            "transport": {"regional": False},
        }
    }

    expected = {
        "energy": {"EU": ["BE,DE,FR"], "global": ["ALL"]},
        "transport": {"NA": ["ALL"]},
    }

    result = setup_alignment_calculator._reconcile_regions()
    
    pd.testing.assert_frame_equal(pd.DataFrame(result), pd.DataFrame(expected))


def test_calculate_tms(setup_alignment_calculator):
    ac = setup_alignment_calculator

    # Run the method under test
    result = ac._calculate_tms(ac._df_climate[2023], "automotive", ac._settings['sectoral_approach']['automotive'], 2023)

    # Check the results
    assert isinstance(result, pd.DataFrame)
    assert ("target" in result.columns)
    assert (result.at[0, "target"] == 90)
    assert (result.at[1, "target"] == 210)


def test_calculate_sda(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_climate = ac._df_climate[2023]
    sector = "coal"
    scenario_year = 2023

    # Run the method under test
    result = ac._calculate_sda(df_climate, sector, scenario_year)
    
    assert isinstance(result, pd.DataFrame)
    assert ("target" in result.columns)
    assert (result.at[0, "target"] == 90)
    assert (result.at[1, "target"] == 180)


@pytest.fixture
def climate_data():
    return pd.DataFrame(
        {
            "company_id": [1, 2, 3],
            "name_company": ["Company A", "Company B", "Company C"],
            "sector": ["Energy", "Energy", "Technology"],
            "technology": ["Coal", "Solar", "AI"],
            "year": [2020, 2020, 2020],
            "region": ["NA", "EU", "APAC"],
            "production": [100, 200, 300],
            "target": [90, 180, 270],
        }
    )


def test_combine_asset_locations(setup_alignment_calculator):
    expected_result = pd.DataFrame(
        {
            "company_id": [88, 110],
            "name_company": ["Company1", "Company2"],
            "sector": ["power", "automotive"],
            "technology": ["power", "automotive"],
            "year": [2023, 2023],
            "region": ["US", "EU"],
            "production": [100, 200],
            "target": [90.0, 180.0],
        }
    )

    result = setup_alignment_calculator._combine_asset_locations(2023)
    assert_frame_equal(result, expected_result)


def test_get_sector_approach_technologies(setup_alignment_calculator):
    ac = setup_alignment_calculator
    expected_technologies = ["electric", "fuelcell", "hybrid", "renewablescap"]
    result_technologies = ac._get_sector_approach_technologies()
    assert set(result_technologies) == set(expected_technologies)


@pytest.fixture
def mock_climate_data():
    return pd.DataFrame(
        {
            "company_id": [1, 2, 3],
            "production": [100, 200, 300],
            "target": [90, 190, 290],
        }
    )


@pytest.fixture
def mock_loan_data():
    return pd.DataFrame(
        {"company_id": [88, 110, 110], "loan_amount": [1000, 2000, 3000]}
    )


def test_combine_climate_loan_data(setup_alignment_calculator):

    result = setup_alignment_calculator._combine_climate_loan_data(
        climate_data=setup_alignment_calculator._df_climate[2023],
        use_loan_file=True,
        individual_loans=False,
        only_parents=True,
        loan_column="outstanding_amount",
        facet_col=[],
        year=2023,
    )

    expected_columns = [
        "company_country",
        "portfolio_code",
        "name_company_x",
        "parent_name",
        "company_lei",
        "company_id",
        "parent_lei",
        "company_name",
        "portfolio_date",
        "outstanding_amount",
        "loan_id",
        "name_company_y",
        "sector",
        "technology",
        "plant_location",
        "year",
        "emission_factor",
        "region",
        "production",
        "target",
    ]

    assert all(col in result.columns for col in expected_columns)
    assert len(result) == 2
    assert all(result["outstanding_amount"].values == [100, 300])


def test_only_parents_true(setup_alignment_calculator):
    ac = setup_alignment_calculator
    result = ac._only_parents(
        ac._loans, "outstanding_amount", 2023, stop_at_weak_parents=True
    )
    expected_result = pd.DataFrame(
        {
            "portfolio_date": [202301, 202301],
            "company_id": [88, 110],
            "loan_id": ["L1", "L2"],
            "portfolio_code": ["portfolio1", "portfolio1"],
            "outstanding_amount": [100, 300],
            "counterparty_id": ["C1", "C2"],
            "company_name": ["Company1", "Company2"],
            "name_company": ["Company1", "Company2"],
            "company_country": ["US", "IT"],
            "company_lei": ["123", "456"],
            "parent_name": ["Company1", "Parent1"],
            "parent_lei": ["123", "789"],
        }
    )
    pd.testing.assert_frame_equal(
        result[
            [
                "portfolio_date",
                "company_id",
                "loan_id",
                "portfolio_code",
                "outstanding_amount",
                "counterparty_id",
                "company_name",
                "name_company",
                "company_country",
                "company_lei",
                "parent_name",
                "parent_lei",
            ]
        ],
        expected_result,
    )


def test_only_parents_false(setup_alignment_calculator):
    ac = setup_alignment_calculator
    result = ac._only_parents(
        ac._loans, "outstanding_amount", 2023, stop_at_weak_parents=False
    )
    expected_result = pd.DataFrame(
        {
            "portfolio_date": [202301, 202301],
            "company_id": [88, 110],
            "loan_id": ["L1", "L2"],
            "portfolio_code": ["portfolio1", "portfolio1"],
            "outstanding_amount": [100, 300],
            "counterparty_id": ["C1", "C2"],
            "company_name": ["Company1", "Company2"],
            "name_company": ["Company1", "Company2"],
            "company_country": ["US", "IT"],
            "company_lei": ["123", "456"],
            "parent_name": ["Company1", "Parent1"],
            "parent_lei": ["123", "789"],
        }
    )
    pd.testing.assert_frame_equal(
        result[
            [
                "portfolio_date",
                "company_id",
                "loan_id",
                "portfolio_code",
                "outstanding_amount",
                "counterparty_id",
                "company_name",
                "name_company",
                "company_country",
                "company_lei",
                "parent_name",
                "parent_lei",
            ]
        ],
        expected_result,
    )


def test_get_parent_companies(setup_alignment_calculator):
    """Test the _get_parent_companies method of alignmentCalculator."""
    ac = setup_alignment_calculator

    # Mocking the _climate_ownership data for the year 2023
    ac._climate_ownership = {
        2023: pd.DataFrame(
            {
                "company_id": ["C1", "C2", "C3"],
                "parent_company_id": ["P1", "P2", "P3"],
                "is_parent": [True, True, True],
                "is_ultimate_listed_parent": [False, True, False],
                "is_ultimate_parent": [True, False, True],
                "ownership_level": [1, 1, 1],
            }
        )
    }

    # Expected DataFrame
    expected_df = pd.DataFrame(
        {"parent_company_id": ["P1", "P2", "P3"]},
        index=pd.Index(["C1", "C2", "C3"], name="company_id"),
    )

    # Run the method
    result_df = ac._get_parent_companies(stop_at_weak_parents=True, year=2023)

    # Check if the result is as expected
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_apply_production_thresholds(setup_alignment_calculator):
    ac = setup_alignment_calculator

    # Create a sample DataFrame that mimics the expected structure
    data = {
        "company_id": ["C1", "C2", "C3", "C4"],
        "sector": ["sector1", "sector1", "sector2", "sector2"],
        "technology": ["tech1", "tech1", "tech2", "tech3"],
        "region": ["EU", "US", "EU", "US"],
        "year": [2023, 2023, 2023, 2023],
        "production": [100, 200, 300, 400],
        "total_assets": [50, 250, 350, 450],
        "turnover": [150, 250, 350, 450],
    }
    loan_data = pd.DataFrame(data)

    # Set production thresholds in the settings
    ac._settings["production_thresholds"] = {
        "sector1": {"asset_ratio": 1, "turnover_ratio": 5},
        "sector2": {"asset_ratio": 0.8, "turnover_ratio": 0.7},
    }

    # Expected DataFrame after applying production thresholds
    expected_data = {
        "company_id": ["C1", "C3", "C4"],
        "sector": ["sector1", "sector2", "sector2"],
        "technology": ["tech1", "tech2", "tech3"],
        "region": ["EU", "EU", "US"],
        "year": [2023, 2023, 2023],
        "production": [100, 300, 400],
        "total_assets": [50, 350, 450],
        "turnover": [150, 350, 450],
    }
    expected_df = pd.DataFrame(expected_data)

    # Apply production thresholds
    result_df = ac._apply_production_thresholds(loan_data)

    # Assert that the resulting DataFrame is as expected
    print(result_df)
    assert_frame_equal(
        result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            pd.DataFrame(
                {
                    "company_id": ["comp1", "comp2", "comp3"],
                    "sector": ["sector1", "sector1", "sector2"],
                    "technology": ["tech1", "tech2", "tech1"],
                    "region": ["US", "EU", "XN"],
                    "year": [2023, 2023, 2023],
                    "total_assets": [100, 200, 300],
                    "turnover": [150, 250, 350],
                    "production": [80, 120, 90],
                }
            ),
            pd.DataFrame(
                {
                    "company_id": ["comp1", "comp2", "comp3"],
                    "sector": ["sector1", "sector1", "sector2"],
                    "total_assets": [100.0, 200.0, 300.0],
                    "turnover": [150.0, 250.0, 350.0],
                    "production": [80, 120, 90],
                    "asset_ratio": [0.8, 0.6, 0.3],
                    "turnover_ratio": [0.5333, 0.48, 0.2571],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "company_id": ["comp1", "comp2"],
                    "sector": ["sector2", "sector2"],
                    "technology": ["tech2", "tech1"],
                    "region": ["US", "XN"],
                    "year": [2023, 2023],
                    "total_assets": [100.0, 200.0],
                    "turnover": [150, 250],
                    "production": [10, 20],
                }
            ),
            pd.DataFrame(
                {
                    "company_id": ["comp1", "comp2"],
                    "sector": ["sector2", "sector2"],
                    "total_assets": [100.0, 200.0],
                    "turnover": [150.0, 250.0],
                    "production": [10, 20],
                    "asset_ratio": [0.1, 0.1],
                    "turnover_ratio": [0.0667, 0.08],
                }
            ),
        ),
    ],
)
def test_determine_ratios(setup_alignment_calculator, test_input, expected):
    ac = setup_alignment_calculator
    result = ac._determine_ratios(test_input)
    pd.testing.assert_frame_equal(result.round(4), expected.round(4))


@pytest.fixture
def sample_loan_data():
    data = {
        "company_id": [88, 110, 88, 110],
        "sector": ["Energy", "Energy", "Transport", "Transport"],
        "production": [100, 200, 300, 400],
        "loan_amount": [1000, 2000, 1500, 2500],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_climate_company_indicators():
    data = {
        "company_id": [88, 110, 88, 110],
        "sector": ["Energy", "Energy", "Transport", "Transport"],
        "production": [120, 180, 320, 380],
    }
    return pd.DataFrame(data)


def test_split_loans_over_sector(
    setup_alignment_calculator, sample_loan_data, sample_climate_company_indicators
):
    setup_alignment_calculator._climate_company_indicators = {
        2023: sample_climate_company_indicators
    }

    expected_data = {
        "company_id": [88, 110, 88, 110],
        "sector": ["Energy", "Energy", "Transport", "Transport"],
        "production": [100, 200, 300, 400],
        "loan_amount": [
            1000 * 120 / 440,
            2000 * 180 / 560,
            1500 * 320 / 440,
            2500 * 380 / 560,
        ],
    }
    expected_df = pd.DataFrame(expected_data)

    result_df = setup_alignment_calculator._split_loans_over_sector(
        sample_loan_data, "loan_amount", 2023
    )
    assert_frame_equal(result_df, expected_df, check_dtype=False)


@pytest.fixture
def sample_climate_data():
    # Sample data simulating the climate data structure
    data = {
        "company_id": ["C1", "C1", "C2"],
        "loan_id": ["L1", "L1", "L2"],
        "portfolio_code": ["p1", "p1", "p1"],
        "sector": ["Energy", "Energy", "Transport"],
        "technology": ["Coal", "Solar", "Electric"],
        "year": [2023, 2023, 2023],
        "portfolio_date": [2023, 2023, 2023],
        "production": [1000, 2000, 1500],
        "target": [900, 1900, 1400],
        "loan_indicator": [100, 200, 150],
    }
    return pd.DataFrame(data)


def test_split_over_technology(setup_alignment_calculator, sample_climate_data):
    # Get the alignmentCalculator instance from the fixture
    calculator = setup_alignment_calculator

    # Call the method under test
    result = calculator._split_over_technology(sample_climate_data, "loan_indicator")

    # Check if the method outputs the expected DataFrame structure
    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."

    # Check if the columns are as expected
    expected_columns = set(
        [
            "company_id",
            "sector",
            "technology",
            "production",
            "target",
            "loan_indicator",
            "year",
            "portfolio_date",
            "portfolio_code",
            "loan_id",
        ]
    )
    assert (
        set(result.columns) == expected_columns
    ), "DataFrame should have specific columns."

    # Check if the technology split logic is correctly applied
    assert any(result["loan_indicator"] != sample_climate_data["loan_indicator"])


def test_normalise_production_global(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_input = pd.DataFrame(
        {
            "company_id": [88, 110],
            "production": [100, 200],
            "outstanding_amount": [40, 50],
            "sector": ["coal", "automotive"],
        }
    )
    expected_output = pd.DataFrame(
        {
            "company_id": [110],
            "production": [200],
            "outstanding_amount": [50],
            "sector": ["automotive"],
            "norm": [
                200
            ],  # Global normalisation based on total production in the sector
        }
    )
    result = ac._normalise_production(
        "global", df_input, 2023, ac._settings["economic_weights"], "outstanding_amount"
    )
    assert_frame_equal(result, expected_output)


def test_normalise_production_economic(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_input = pd.DataFrame(
        {
            "company_id": [88, 110],
            "production": [100, 200],
            "sector": ["coal", "automotive"],
        }
    )
    expected_output = pd.DataFrame(
        {
            "company_id": [110],
            "production": [200],
            "sector": ["automotive"],
            "norm": [34554000],  # Economic weights applied
        }
    )
    result = ac._normalise_production(
        "economic", df_input, 2023, ac._settings["economic_weights"], "production"
    )
    assert_frame_equal(result, expected_output)


def test_global_normalisation(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    # Assume _global_normalisation is a method of alignmentCalculator
    # This is a hypothetical test; the actual implementation details might differ
    normalised_data = ac._global_normalisation(sample_data, 2023)

    # Check if the normalisation column exists
    assert "norm" in normalised_data.columns, "Normalization column 'norm' should exist"

    sample_data.loc[1, "production"] = 100
    assert all(normalised_data["norm"].values == sample_data["production"].values)


@pytest.fixture
def economic_weights():
    return {"sector1": 1.5, "sector2": 0.5}


@pytest.fixture
def df_combined():
    # Mock data simulating combined DataFrame after processing
    return pd.DataFrame(
        {
            "company_id": [88, 88, 110, 110],
            "sector": ["coal", "automotive", "power", "coal"],
            "production": [100, 200, 300, 400],
            "year": [2023, 2023, 2023, 2023],
        }
    )


def test_economic_normalisation(setup_calculator, df_combined, economic_weights):
    # Expected DataFrame after applying economic normalisation
    expected_df = pd.DataFrame(
        {
            "company_id": [88, 110, 88, 110],
            "sector": ["coal", "coal", "automotive", "power"],
            "production": [100, 400, 200, 300],
            "year": [2023, 2023, 2023, 2023],
            "norm": [58103, 58103, 3484572, 71606],  # Normalised by economic weights
        }
    )

    # Run the economic normalisation method
    result_df = setup_calculator._economic_normalisation(
        df_combined, economic_weights, 2023
    )
    # Assert that the resulting DataFrame is as expected
    assert_frame_equal(result_df, expected_df)


@pytest.fixture
def setup_calculator():
    ac = alignmentCalculator()
    return ac


def test_total_normalisation(setup_calculator):
    ac = setup_calculator
    # Create a sample DataFrame to simulate combined data
    data = pd.DataFrame(
        {
            "year": [2023, 2023, 2023],
            "sector": ["Energy", "Energy", "Transport"],
            "production": [100, 200, 300],
            "loan_indicator": [1000, 2000, 1500],
        }
    )
    loan_indicator = "loan_indicator"

    # Expected normalization should sum production for each sector and use it as norm
    expected_norms = pd.Series([300, 300, 300], name="norm")

    # Run the total_normalisation method
    normalized_data = ac._total_normalisation(data, loan_indicator, 2023)

    # Check if the 'norm' column matches the expected values
    pd.testing.assert_series_equal(normalized_data["norm"], expected_norms)


def test_portfolio_normalisation(setup_alignment_calculator):
    ac = setup_alignment_calculator
    # Create a sample DataFrame to simulate combined data
    data = pd.DataFrame(
        {
            "year": [2023, 2023, 2023, 2023],
            "company_id": [88, 110, 88, 110],
            "sector": ["coal", "coal", "automotive", "power"],
            "production": [100, 200, 300, 400],
            "loan_indicator": [1000, 2000, 3000, 4000],
            "portfolio_code": ["A", "A", "B", "B"],
        }
    )

    # Expected DataFrame after normalisation
    expected_data = pd.DataFrame(
        {
            "year": [2023, 2023, 2023, 2023],
            "company_id": [88, 110, 88, 110],
            "sector": ["coal", "coal", "automotive", "power"],
            "production": [100, 200, 300, 400],
            "loan_indicator": [1000, 2000, 3000, 4000],
            "portfolio_code": ["A", "A", "B", "B"],
            "norm": [
                300,
                300,
                300,
                400,
            ],  # Normalisation factor based on sector and portfolio_id
        }
    )

    # Run the normalisation method
    result_data = ac._portfolio_normalisation(data, "loan_indicator", 2023)
    print(result_data)
    # Check if the resulting DataFrame is as expected
    assert_frame_equal(result_data, expected_data)


def test_company_normalisation(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    # Assume _company_normalisation is a method of alignmentCalculator
    # and it has been implemented to normalize 'production' based on 'loan_amount'

    # Expected DataFrame after normalization
    expected_data = {
        "company_id": ["C1", "C2", "C3"],
        "name_company": ["Company1", "Company2", "Company3"],
        "sector": ["power", "power", "automotive"],
        "technology": ["coalcap", "oilcap", "electric"],
        "plant_location": ["US", "CA", "DE"],
        "year": [2023, 2023, 2023],
        "production": [100, 150, 200],
        "emission_factor": [0.1, 0.2, 0.3],
        "region": ["North America", "North America", "Europe"],
        "target": [90, 140, 190],
        "norm": [
            100,
            150,
            200,
        ],
    }
    expected_df = pd.DataFrame(expected_data)

    # Run the normalization method
    result_df = ac._company_normalisation(sample_data)
    print(result_df)
    # Check if the resulting DataFrame is as expected
    assert_frame_equal(result_df, expected_df)


def test_calculate_alignment_instance(setup_alignment_calculator):
    ac = setup_alignment_calculator
    data = pd.DataFrame(
        {
            "company_id": [88, 110],
            "portfolio_code": ["portfolio1", "portfolio1"],
            "counterparty_id": ["C1", "C2"],
            "company_name": ["Company1", "Company2"],
            "name_company": ["Company1", "Company2"],
            "company_country": ["US", "IT"],
            "company_lei": ["123", "456"],
            "parent_name": ["Company1", "Parent1"],
            "parent_lei": ["123", "789"],
            "company_id": [1, 2],
            "loan_id": ["L1", "L2"],
            "portfolio_date": [202301, 202301],
            "outstanding_amount": [100, 300],
            "technology": ["coalcap", "electric"],
            "sector": ["power", "automotive"],
            "year": [2023, 2023],
            "production": [100, 200],
            "target": [90, 180],
            "region": ["US", "EU"],
        }
    )
    facet_col = []
    loan_indicator = "outstanding_amount"
    bopo_split = False
    limit = 3

    # Call the method under test
    result = ac._calculate_alignment_instance(
        data, facet_col, loan_indicator, bopo_split, limit, 0
    )
    # Check the results
    expected = pd.DataFrame(
        {
            "portfolio_code": ["portfolio1"],
            "portfolio_date": [202301],
            "end_year": [2023],
            "outstanding_amount": [400],
            "weighted_deviation": [5000],
            "weighted_target": [63000],
            "score": [0.079365],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_make_weighted_target(setup_alignment_calculator):
    ac = setup_alignment_calculator
    data = pd.DataFrame(
        {
            "company_id": [1, 2],
            "year": [2023, 2023],
            "end_year": [2025, 2025],
            "production": [1000, 2000],
            "target": [900, 1800],
            "loan_indicator": [100, 200],
            "technology": ["coal", "coal"],
            "sector": ["coal", "coal"],
            "deviation": [100, 200],
        }
    )
    cols = ["company_id", "year"]
    end_year = 2025
    result = ac._make_weighted_target(data, "loan_indicator", cols, end_year)
    expected = pd.DataFrame(
        {
            "company_id": [1, 2],
            "year": [2023, 2023],
            "end_year": [2025, 2025],
            "production": [1000, 2000],
            "target": [900, 1800],
            "loan_indicator": [100, 200],
            "technology": ["coal", "coal"],
            "sector": ["coal", "coal"],
            "deviation": [-100, -200],
            "weighted_deviation": [-10000, -40000],
            "target_end": [900, 1800],
            "weighted_target": [90000, 360000],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_make_bopo_split(setup_alignment_calculator, sample_data):
    # Mocking the settings to include necessary sectoral approach details
    setup_alignment_calculator._settings = {
        "sectoral_approach": {
            "power": {
                "build_out": ["renewablecap"],
                "phase_out": ["coalcap"],
                "other": [],
            },
            "coal": {"build_out": [], "phase_out": ["coal"], "other": []},
            "automotive": {
                "build_out": ["electric"],
                "phase_out": ["ice"],
                "other": [],
            },
        }
    }

    # Expected output
    expected_directions = ["phase_out", "no_change", "build_out"]

    # Running the method
    result = setup_alignment_calculator._make_bopo_split(sample_data)

    # Checking if the directions are correctly assigned
    assert list(result["direction"]) == expected_directions


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
        "portfolio_id": ["test_portfolio", "test_portfolio", "test_portfolio"],
    }
    expected_df = pd.DataFrame(expected_data)

    # Running the actual aggregation method
    results = ac._calculate_alignment_instance(
        data=sample_data,
        facet_col=["sector", "technology", "region"],
        loan_indicator="production",
        bopo_split=False,
        limit=3,
        horzion=0,
    )

    # Asserting if the results are as expected
    assert_frame_equal(results, expected_df, check_dtype=False)


def test_make_portfolio_dates(setup_alignment_calculator):
    ac = setup_alignment_calculator
    ac._loans["portfolio_date"] = [202312, 202311]
    expected_dates = {202312}
    assert ac._make_portfolio_dates() == expected_dates


@patch("alignment_calculation.calculator.alignmentCalculator._preprocess_data")
def test_make_master_data(
    mock_preprocess_data_func,
    setup_alignment_calculator,
    loan_indicator,
    only_parents,
    facet_col,
    use_region_file,
    normalise_method,
):
    mock_preprocess_data_func.return_value = pd.DataFrame(
        {
            "company_id": [1, 2],
            "loan_indicator": [100, 200],
            "facet_col": ["sector1", "sector2"],
            "use_region_file": [True, True],
            "normalise_method": ["total", "total"],
        }
    )

    result = setup_alignment_calculator._make_master_data(
        loan_indicator=loan_indicator,
        only_parents=only_parents,
        facet_col=facet_col,
        use_region_file=use_region_file,
        normalise_method=normalise_method,
    )

    assert isinstance(result, dict)
    assert all(isinstance(v, pd.DataFrame) for v in result[2023].values())
    assert all("company_id" in df.columns for df in result[2023].values())
    assert all("loan_indicator" in df.columns for df in result[2023].values())
    assert all("facet_col" in df.columns for df in result[2023].values())
    assert all("use_region_file" in df.columns for df in result[2023].values())
    assert all("normalise_method" in df.columns for df in result[2023].values())
    mock_preprocess_data_func.assert_called()


def test_aggregate_over_time_results(setup_alignment_calculator, sample_data):
    with patch.object(
        setup_alignment_calculator,
        "_make_time_metrics",
        return_value=(sample_data, sample_data),
    ):
        result = setup_alignment_calculator._aggregate_over_time_results(
            sample_data, "loan_indicator", add_total=True
        )
        assert not result.empty, "The result should not be empty."
        assert (
            "total_shift" in result.columns
        ), "The result should have a 'total_shift' column."
        assert (
            result["total_shift"].notna().all()
        ), "All values in 'total_shift' should be non-NaN."
        assert result["score"].notna().all(), "All values in 'score' should be non-NaN."
        assert (result["score"] <= 3).all() and (
            result["score"] >= -3
        ).all(), "Scores should be clipped between -3 and 3."
