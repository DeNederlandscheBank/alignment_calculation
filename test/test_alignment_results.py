from unittest.mock import patch
import pandas as pd
import pytest
import os.path as path
from alignment_calculation.alignment_results import alignmentResults


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "portfolio_id": [1, 2, 3],
            "loan_id": [4, 5, 6],
            "company_id": [1, 2, 3],
            "end_year": [2022, 2022, 2022],
            "weighted_deviation": [10, 20, 30],
            "weighted_target": [100, 200, 300],
            "loan_amount": [10, 20, 30],
            "sector": ["coal", "power", "steel"],
            "technology": ["coal", "oilcap", "coalcap"],
            "portfolio_date": ["2022-01-01", "2023-01-01", "2024-01-01"],
        }
    )


@pytest.fixture
def settings():
    package_dir = path.dirname(path.dirname(path.abspath(__file__)))
    return {
        "company_information_file": path.join(
            package_dir,
            "alignment_calculation",
            "data",
            "company_data",
            "company_information.csv",
        ),
        "sectoral_approach": {
            "power": {"approach": "sda", "other": ["oilcap", "coalcap"]}
        },
    }


@pytest.fixture
def climate_company_indicators():
    return {
        "2022": pd.DataFrame(
            {
                "company_id": [1, 2, 3],
                "sector": ["coal", "power", "steel"],
                "technology": ["coal", "oilcap", "coalcap"],
                "year": [2022, 2022, 2022],
                "production": [1000, 2000, 3000],
                "plant_location": ["US", "DE", "CN"],
            }
        )
    }


@pytest.fixture
def df_climate():
    return {
        "2022": pd.DataFrame(
            {
                "company_id": [1, 2, 3],
                "year": [2022, 2022, 2022],
                "sector": ["coal", "power", "steel"],
                "technology": ["coal", "oilcap", "coalcap"],
                "name_company": ["Company A", "Company B", "Company C"],
                "target": [1100, 1900, 3000],
            }
        )
    }


@pytest.fixture
def scenario_data():
    return {
        "2022": pd.DataFrame(
            {
                "company_id": [1, 2, 3],
                "year": [2022, 2022, 2022],
                "sector": ["coal", "power", "steel"],
                "technology": ["coal", "oilcap", "coalcap"],
                "target": [1500, 2500, 3500],
            }
        )
    }


@pytest.fixture
def alignment_results_instance(
    sample_data, climate_company_indicators, df_climate, scenario_data, settings
):
    return alignmentResults(
        results_data=sample_data,
        climate_company_indicators=climate_company_indicators,
        df_climate=df_climate,
        scenario_data=scenario_data,
        settings=settings,
        portfolio_id="portfolio_id",
    )


def test_get_results(alignment_results_instance):
    results = alignment_results_instance.get_results()
    assert "weighted_deviation" not in results.columns
    assert "weighted_target" not in results.columns


def test_group_scores(alignment_results_instance):
    grouped = alignment_results_instance.group_scores(grouper=["company_id"])
    assert "score" in grouped.columns
    assert grouped["score"].iloc[0] == 0.1 


def test_add_information_to_results(alignment_results_instance):

    updated = alignment_results_instance.add_information_to_results("loan_amount")

    assert "sector_main" in updated.columns
    assert "name_company" in updated.columns
    assert "domicile" in updated.columns
    assert "plant_location" in updated.columns
    assert "production" in updated.columns
    assert "target" in updated.columns


def test_add_company_names(alignment_results_instance):
    with_names = alignment_results_instance._add_company_names(
        alignment_results_instance._results_data
    )
    assert "name_company" in with_names.columns


def test_add_main_sector(alignment_results_instance):
    with_sector = alignment_results_instance._add_main_sector(
        alignment_results_instance._results_data, "loan_amount"
    )
    assert "sector_main" in with_sector.columns


def test_add_production(alignment_results_instance):
    with_production = alignment_results_instance._add_production(
        alignment_results_instance._results_data
    )
    assert "production" in with_production.columns


def test_add_company_domicile(alignment_results_instance):
    with_domicile = alignment_results_instance._add_company_domicile(
        alignment_results_instance._results_data
    )
    assert "domicile" in with_domicile.columns


def test_add_production_location(alignment_results_instance):
    with_location = alignment_results_instance._add_production_location(
        alignment_results_instance._results_data, "loan_amount"
    )
    assert "plant_location" in with_location.columns


def test_add_target(alignment_results_instance):
    with_target = alignment_results_instance._add_target(alignment_results_instance._results_data)
    assert 'target' in with_target.columns
