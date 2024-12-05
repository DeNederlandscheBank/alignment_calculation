import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock
from alignment_calculation.calculator import alignmentCalculator


@pytest.fixture
def sample_data():
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


@pytest.fixture(autouse=True)
def setup_alignment_calculator():
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


def test_default_initialization():
    default_alignment_calculator = alignmentCalculator(portfolio_id="portfolio_code")
    assert default_alignment_calculator._portfolio_id == "portfolio_code"
    assert default_alignment_calculator._scenario_set == "weo"
    assert default_alignment_calculator._pathway == "nze_2050"
    assert default_alignment_calculator._settings is not None
    assert not hasattr(default_alignment_calculator, "_loans")


def test_custom_initialization():
    custom_alignment_calculator = alignmentCalculator(
        portfolio_id="custom_portfolio",
        custom_settings={
            "company_information_file": "../data/company_data/company_information.csv"
        },
        loan_file=pd.DataFrame({"column": [1, 2, 3]}),
        loanbook_settings={"base_year": 2022, "external_columns": None},
        scenario_set="custom_scenario",
        pathway="custom_pathway",
        debug=False,
    )
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


def test_get_available_scenarios(setup_alignment_calculator):
    expected_df = pd.DataFrame(
        {
            "scenario_set": ["weo"],
            "pathway": ["nze_2050"],
            "year": [2023],
        }
    )

    result_df = setup_alignment_calculator.get_available_scenarios()

    pd.testing.assert_frame_equal(
        result_df.sort_values(by="year").reset_index(drop=True),
        expected_df.sort_values(by="year").reset_index(drop=True),
    )


def test_set_scenarios(setup_alignment_calculator):
    setup_alignment_calculator._scenario_data = {
        2022: {"weo": {"nze_2050": MagicMock()}}
    }

    scenario_set = "weo"
    pathway = "nze_2050"
    setup_alignment_calculator.set_scenarios(scenario_set, pathway)

    assert setup_alignment_calculator._scenario_set == scenario_set
    assert setup_alignment_calculator._pathway == pathway


def test_update_settings(setup_alignment_calculator):
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

        new_settings = {"company_information_file": "new_value"}
        setup_alignment_calculator.update_settings(new_settings)

        mock_config.config.assert_called_once_with(
            main_climate_file=None,
            company_information_file="new_value",
            economic_weights=None,
            production_thresholds=None,
            scenario_data=None,
            sectoral_approach=None,
            save_changes=True,
        )

        assert (
            setup_alignment_calculator._settings["company_information_file"]
            == "original_value"
        )

        assert mock_config.load_settings.not_called


def test_update_loanbook_with_file(setup_alignment_calculator):
    with patch(
        "alignment_calculation.calculator._load_loanbook_data",
        return_value=pd.DataFrame(
            {"company_id": [1, 2, 3], "loan_amount": [1000, 2000, 3000]}
        ),
    ) as mock_load:
        setup_alignment_calculator.update_loanbook(loan_file="test_loans.csv")
        mock_load.assert_called_once_with("test_loans.csv")
        assert isinstance(setup_alignment_calculator._loans, pd.DataFrame)
        assert not setup_alignment_calculator._loans.empty
        assert setup_alignment_calculator._loans.equals(
            pd.DataFrame({"company_id": [1, 2, 3], "loan_amount": [1000, 2000, 3000]})
        )


def test_update_loanbook_with_settings(setup_alignment_calculator):
    loanbook_settings = {
        "base_year": 2023,
        "month": 1,
        "loan_file": "test_loans.csv",
        "additional_instrument_columns": {"column1": "value1", "column2": "value2"},
    }
    with patch("alignment_calculation.calculator.loanbookPreparer") as mock_preparer:
        preparer_instance = mock_preparer.return_value
        preparer_instance.prepare_loanbook.return_value = pd.DataFrame(
            {"company_id": [1, 2, 3], "loan_amount": [1000, 2000, 3000]}
        )
        setup_alignment_calculator.update_loanbook(loanbook_settings=loanbook_settings)
        preparer_instance.prepare_loanbook.assert_called_once_with(**loanbook_settings)
        assert isinstance(setup_alignment_calculator._loans, pd.DataFrame)
        assert not setup_alignment_calculator._loans.empty


@patch("alignment_calculation.calculator._load_region_data")
@patch("alignment_calculation.calculator.alignmentCalculator._calculate_climate")
@patch("alignment_calculation.calculator.alignmentCalculator._combine_asset_locations")
@patch(
    "alignment_calculation.calculator.alignmentCalculator._combine_climate_loan_data"
)
@patch(
    "alignment_calculation.calculator.alignmentCalculator._apply_production_thresholds"
)
@patch("alignment_calculation.calculator.alignmentCalculator._split_loans_over_sector")
@patch("alignment_calculation.calculator.alignmentCalculator._split_over_technology")
@patch("alignment_calculation.calculator.alignmentCalculator._normalise_production")
def test_preprocess_data(
    mock_normalise_production,
    mock_split_over_technology,
    mock_split_loans_over_sector,
    mock_apply_production_thresholds,
    mock_combine_climate_loan_data,
    mock_combine_asset_locations,
    mock_calculate_climate,
    mock_load_region_data,
    setup_alignment_calculator,
):
    mock_normalise_production.return_value = pd.DataFrame(
        {
            "target": [10, 20, 30, 40, 50],
            "production": [10, 20, 30, 40, 50],
            "norm": [5, 4, 3, 2, 1],
        }
    )
    mock_split_over_technology.return_value = pd.DataFrame()
    mock_split_loans_over_sector.return_value = pd.DataFrame()
    mock_apply_production_thresholds.return_value = pd.DataFrame()
    mock_combine_climate_loan_data.return_value = pd.DataFrame()
    mock_combine_asset_locations.return_value = pd.DataFrame()
    mock_calculate_climate.return_value = pd.DataFrame()
    mock_load_region_data.return_value = pd.DataFrame()

    result = setup_alignment_calculator._preprocess_data(
        use_loan_file=True,
        individual_loans=False,
        loan_indicator="outstanding_amount",
        only_parents=True,
        facet_col=["sector"],
        use_region_file=True,
        year=2023,
        normalise_method="total",
    )

    assert not result.empty
    assert "target" in result.columns
    mock_normalise_production.assert_called_once()
    mock_split_over_technology.assert_called_once()
    mock_split_loans_over_sector.assert_called_once()
    mock_apply_production_thresholds.assert_called_once()
    mock_combine_climate_loan_data.assert_called_once()
    mock_combine_asset_locations.assert_called_once()
    mock_calculate_climate.assert_called_once()
    mock_load_region_data.assert_called_once()


def test_add_region(setup_alignment_calculator):
    calculator = setup_alignment_calculator
    year = 2023
    region_mapping = {
        "power": {"region1": ["US"]},
        "automotive": {"region2": ["US"]},
    }
    calculator._add_region(year, region_mapping)

    assert calculator._climate_company_indicators[2023]["region"].iloc[0] == "region1"
    assert calculator._climate_company_indicators[2023]["region"].iloc[1] == "global"


def test_reconcile_regions_no_region_data(setup_alignment_calculator):
    """Test _reconcile_regions method when no region data is loaded."""
    setup_alignment_calculator._regions = None
    result = setup_alignment_calculator._reconcile_regions()
    assert result is None, "Expected None when no region data is available"


def test_reconcile_regions_with_region_data(setup_alignment_calculator):
    """Test _reconcile_regions method with mock region data."""
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

    result = ac._calculate_tms(
        ac._df_climate[2023],
        "automotive",
        ac._settings["sectoral_approach"]["automotive"],
        2023,
    )

    assert isinstance(result, pd.DataFrame)
    assert "target" in result.columns
    assert result.at[0, "target"] == 90
    assert result.at[1, "target"] == 210


def test_calculate_sda(setup_alignment_calculator):
    ac = setup_alignment_calculator
    df_climate = ac._df_climate[2023]
    sector = "coal"
    scenario_year = 2023

    result = ac._calculate_sda(df_climate, sector, scenario_year)

    assert isinstance(result, pd.DataFrame)
    assert "target" in result.columns
    assert result.at[0, "target"] == 90
    assert result.at[1, "target"] == 180


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
    result = result.sort_values(by=["company_id"])
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

    expected_df = pd.DataFrame(
        {"parent_company_id": ["P1", "P2", "P3"]},
        index=pd.Index(["C1", "C2", "C3"], name="company_id"),
    )

    result_df = ac._get_parent_companies(stop_at_weak_parents=True, year=2023)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_apply_production_thresholds(setup_alignment_calculator):
    ac = setup_alignment_calculator

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

    ac._settings["production_thresholds"] = {
        "sector1": {"asset_ratio": 1, "turnover_ratio": 5},
        "sector2": {"asset_ratio": 0.8, "turnover_ratio": 0.7},
    }

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

    result_df = ac._apply_production_thresholds(loan_data)

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


def test_split_loans_over_sector(setup_alignment_calculator):
    setup_alignment_calculator._climate_company_indicators = {
        2023: pd.DataFrame(
            {
                "company_id": [88, 110, 88, 110],
                "sector": ["Energy", "Energy", "Transport", "Transport"],
                "production": [120, 180, 320, 380],
            }
        )
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
        pd.DataFrame(
            {
                "company_id": [88, 110, 88, 110],
                "sector": ["Energy", "Energy", "Transport", "Transport"],
                "production": [100, 200, 300, 400],
                "loan_amount": [1000, 2000, 1500, 2500],
            }
        ),
        "loan_amount",
        2023,
    )
    assert_frame_equal(result_df, expected_df, check_dtype=False)


def test_split_over_technology(setup_alignment_calculator):
    calculator = setup_alignment_calculator

    climate_data = pd.DataFrame(
        {
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
    )
    result = calculator._split_over_technology(climate_data, "loan_indicator")

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame."

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

    assert any(result["loan_indicator"] != climate_data["loan_indicator"])


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
            "norm": [200],
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
            "norm": [34554000],
        }
    )
    result = ac._normalise_production(
        "economic", df_input, 2023, ac._settings["economic_weights"], "production"
    )
    assert_frame_equal(result, expected_output)


def test_global_normalisation(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    normalised_data = ac._global_normalisation(sample_data, 2023)

    assert "norm" in normalised_data.columns, "Normalization column 'norm' should exist"

    sample_data.loc[1, "production"] = 100
    assert all(normalised_data["norm"].values == sample_data["production"].values)


def test_economic_normalisation(setup_alignment_calculator):
    expected_df = pd.DataFrame(
        {
            "company_id": [88, 110],
            "sector": ["automotive", "power"],
            "production": [100, 400],
            "year": [2023, 2023],
            "norm": [400, 10],
        }
    )

    result_df = setup_alignment_calculator._economic_normalisation(
        pd.DataFrame(
            {
                "company_id": [88, 110],
                "sector": ["automotive", "power"],
                "production": [100, 400],
                "year": [2023, 2023],
                "norm": [400, 10],
            }
        ),
        {"automotive": 2, "power": 0.1, "coal": 0.5},
        2023,
    )
    print(result_df)
    assert_frame_equal(result_df, expected_df)


def test_total_normalisation():
    ac = alignmentCalculator()
    data = pd.DataFrame(
        {
            "year": [2023, 2023, 2023],
            "sector": ["Energy", "Energy", "Transport"],
            "production": [100, 200, 300],
            "loan_indicator": [1000, 2000, 1500],
        }
    )
    loan_indicator = "loan_indicator"

    expected_norms = pd.Series([300, 300, 300], name="norm")

    normalized_data = ac._total_normalisation(data, loan_indicator, 2023)

    pd.testing.assert_series_equal(normalized_data["norm"], expected_norms)


def test_portfolio_normalisation(setup_alignment_calculator):
    ac = setup_alignment_calculator
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
            ],
        }
    )

    result_data = ac._portfolio_normalisation(data, "loan_indicator", 2023)
    print(result_data)
    assert_frame_equal(result_data, expected_data)


def test_company_normalisation(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
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

    result_df = ac._company_normalisation(sample_data)
    print(result_df)
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

    result = ac._calculate_alignment_instance(
        data, facet_col, loan_indicator, bopo_split, limit, 0
    )
    expected = pd.DataFrame(
        {
            "end_year": [2023],
            "portfolio_code": ["portfolio1"],
            "portfolio_date": [202301],
            "outstanding_amount": [400],
            "weighted_deviation": [5000],
            "weighted_target": [63000],
            "score": [0.079365],
        }
    )

    pd.testing.assert_frame_equal(
        result[
            [
                "end_year",
                "portfolio_code",
                "portfolio_date",
                "outstanding_amount",
                "weighted_deviation",
                "weighted_target",
                "score",
            ]
        ],
        expected,
    )


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
    expected_directions = ["phase_out", "no_change", "build_out"]

    result = setup_alignment_calculator._make_bopo_split(sample_data)

    assert list(result["direction"]) == expected_directions


def test_aggregate_results(setup_alignment_calculator, sample_data):
    ac = setup_alignment_calculator
    data = pd.DataFrame(
        {
            "weighted_deviation": [100, 200, 300, 400],
            "weighted_target": [50, 60, 70, 80],
            "id": ["a", "b", "c", "a"],
            "loan_indicator": [400, 300, 200, 100],
        }
    )
    expected_data = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "loan_indicator": [500, 300, 200],
            "weighted_deviation": [500, 200, 300],
            "weighted_target": [130, 60, 70],
            "score": [3.0, 3.0, 3.0],
        }
    )

    results = ac._aggregate_results(data, ["id"], "loan_indicator", 3)
    assert_frame_equal(results, expected_data)


def test_make_portfolio_dates(setup_alignment_calculator):
    ac = setup_alignment_calculator
    ac._loans["portfolio_date"] = [202312, 202311]
    expected_dates = {202312}
    assert ac._make_portfolio_dates() == expected_dates


@patch("alignment_calculation.calculator.alignmentCalculator._preprocess_data")
def test_make_master_data(mock_preprocess_data_func, setup_alignment_calculator):
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
        loan_indicator="loan_indicator",
        only_parents=False,
        facet_col=[],
        use_region_file=True,
        normalise_method="total",
    )

    assert isinstance(result, dict)
    assert all(isinstance(v, pd.DataFrame) for v in result[2023].values())
    assert all("company_id" in df.columns for df in result[2023].values())
    assert all("loan_indicator" in df.columns for df in result[2023].values())
    assert all("facet_col" in df.columns for df in result[2023].values())
    assert all("use_region_file" in df.columns for df in result[2023].values())
    assert all("normalise_method" in df.columns for df in result[2023].values())
    mock_preprocess_data_func.assert_called()
