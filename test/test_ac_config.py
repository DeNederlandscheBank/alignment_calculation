import pytest
from unittest.mock import patch, mock_open, MagicMock
from alignment_calculation.ac_config import alignmentCalculatorConfig


@pytest.fixture
def config_instance():
    return alignmentCalculatorConfig()


def test_config_save_changes_true(config_instance):
    with (
        patch.object(config_instance, "load_settings", return_value={
            "main_climate_file": {},
            "company_information_file": "",
            "economic_weights": {},
            "production_thresholds": {},
            "scenario_data": {},
            "sectoral_approach": {},}),
        patch.object(config_instance, "_save_settings") as mock_save,
    ):
        new = config_instance.config(scenario_data={"sector": "coal"}, save_changes=True)
        mock_save.assert_called_once()
        assert new is None


def test_config_save_changes_false(config_instance):
    with (
        patch.object(config_instance, "load_settings", return_value={
            "main_climate_file": {},
            "company_information_file": "",
            "economic_weights": {},
            "production_thresholds": {},
            "scenario_data": {},
            "sectoral_approach": {},}),
        patch.object(config_instance, "_save_settings") as mock_save,
    ):
        result = config_instance.config(save_changes=False)
        assert isinstance(result, dict)
        mock_save.assert_not_called()


def test_adjust_path(config_instance):
    with patch("os.path.dirname", return_value="/home/user"):
        result = config_instance._adjust_path("..")
        assert (result == "/home/user") or (result == "\\home\\user")


def test_replace_in_nested_dict_with_dict(config_instance):
    test_dict = {"path": ".."}
    with patch.object(config_instance, "_adjust_path", return_value="/home/user"):
        result = config_instance._replace_in_nested_dict(test_dict)
        assert result == {"path": "/home/user"}


def test_replace_in_nested_dict_with_list(config_instance):
    test_list = [".."]
    with patch.object(config_instance, "_adjust_path", return_value="/home/user"):
        result = config_instance._replace_in_nested_dict(test_list)
        assert result == ["/home/user"]


def test_load_and_check_setting_existing(config_instance):
    old_settings = {"test_setting": "value"}
    result = config_instance._load_and_check_setting("test_setting", old_settings)
    assert result == "value"


def test_load_and_check_setting_new(config_instance):
    old_settings = {}
    new_setting = "new_value"
    result = config_instance._load_and_check_setting(
        "test_setting", old_settings, new_setting
    )
    assert result == "new_value"


def test_reset_settings(config_instance):
    with (
        patch.object(config_instance, "load_settings", return_value={}),
        patch.object(config_instance, "_save_settings") as mock_save,
    ):
        config_instance.reset_settings()
        mock_save.assert_called_once()


def test_load_settings_original_false(config_instance):
    with (
        patch("os.path.isfile", return_value=True),
        patch("ruamel.yaml.YAML") as mock_yaml,
        patch("builtins.open", mock_open(read_data="data")),
    ):
        mock_yaml_instance = mock_yaml.return_value
        mock_yaml_instance.load.return_value = {}
        result = config_instance.load_settings()
        assert isinstance(result, dict)


def test_load_settings_original_true(config_instance):
    with (
        patch("os.path.isfile", return_value=False),
        patch("ruamel.yaml.YAML") as mock_yaml,
        patch("builtins.open", mock_open(read_data="data")),
    ):
        mock_yaml_instance = mock_yaml.return_value
        mock_yaml_instance.load.return_value = {}
        result = config_instance.load_settings(original_settings=True)
        assert isinstance(result, dict)
