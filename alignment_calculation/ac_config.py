from typing import Any
import ruamel.yaml 
import os


class alignmentCalculatorConfig:

    def __init__(self):
        pass

    def config(
        self,
        main_climate_file: dict|None = None,
        company_information_file: str|None = None,
        economic_weights: dict|None = None,
        production_thresholds: dict|None = None,
        scenario_data: dict|None = None,
        sectoral_approach: dict|None = None,
        save_changes: bool|None = True,
    ) -> None| dict:
        """Configure the settings for the alignment calculator.

        The adjusted settings will be saved in the .climate folder in the userdir.
        The settings are in the form of a YAML file and can also be edited manually.

        Parameters
        ----------
        main_climate_file : dict|None, optional
            Dictionary containing main climate file settings.
            The dict should adhere to the following format
            {year: [file_location_company_analytics, file_location_company_indicators],
            ....}
            default= None
        company_information_file : str|None, optional
            Absolute path to the company information file as a string
            default= None
        economic_weights : dict|None, optional
            Dictionary containing economic weights, containing a weight for each sector
            {'power':800, ...}
            default= None
        production_thresholds : dict|None, optional
            Dictionary containing production thresholds, containing both a
            asset_ratio and a turnover_ratio for each sector
            {'power':{'asset_ratio':0.0003, 'turnover_ratio':0.001}, ...}
            default= None
        scenario_data: dict|None, optional
            Dictionary containing scenario files, both the scenario data and the region
            data. The dict should be in the following form:
            {'scenario_files':
                {year:
                    {scenario_set:
                        {'scenario_file_tms': file_location_scenario_file_tms,
                        'scenario_file_sda': file_location_scenario_file_sda}
                    ,...}
                ,...},
                'region_file':
                {scenario_set: file_location_scenario_region_file, ...}
            }
            default= None
        sectoral_approach : dict|None, optional
            Dictionary containing sectoral approach settings.
            default= None
        save_changes: bool|None, optional
            Flag whether the changes to the settings should be saved to file,
            if True changes are saved to file, if False new settings are returned.
            default= True

        Returns
        -------
        None| dict
            the dict with the new settings, or if thesave_changes book is True
            the changes are saved to file and None is returned.

        Notes
        -----
        This function loads existing settings, updates them with provided new settings,
        and then saves the updated settings. If a parameter is None, the corresponding
        setting is not updated and remains unchanged. All paths provided in the settings
        should be absolute paths.
        """
        changes = {
            "main_climate_file": main_climate_file,
            "company_information_file": company_information_file,
            "economic_weights": economic_weights,
            "production_thresholds": production_thresholds,
            "scenario_data": scenario_data,
            "sectoral_approach": sectoral_approach,
        }
        old_settings = self.load_settings()
        new_settings = dict()

        for option, change in changes.items():
            new_settings[option] = self._load_and_check_setting(
                option, old_settings, change
            )

        if save_changes:
            self._save_settings(new_settings)
        else:
            return new_settings

    def _adjust_path(self, relative_path: str) -> str:
        """Adjust a relative path to an absolute path.

        This function takes a relative path (`relative_path`) and adjusts it to
        an absolute path based on the directory of the script where this function
        is called. It replaces occurrences of `..` in the path with the directory
        containing the script.

        Parameters
        ----------
        relative_path : str
            The relative path to be adjusted to an absolute path.

        Returns
        -------
        str
            The adjusted absolute path.

        Notes
        -----
        This function replaces occurrences of `..` in the relative path with the
        directory containing the script (`__file__`). It then normalizes the path
        using `os.path.normpath` to handle any redundant separators and references.
        """
        absolute_path = relative_path.replace("..", os.path.dirname(__file__))
        absolute_path = os.path.normpath(absolute_path)
        absolute_path = absolute_path.replace(":/", "://")

        return absolute_path

    def _replace_in_nested_dict(
        self, dict_data: dict| str| list
    ) -> dict| str| list:
        """Recursively replace relative paths with absolute paths in a nested dictionary.

        This function traverses through a nested dictionary and replaces occurrences of
        relative paths with absolute paths. It supports replacing paths in (nested)
        dictionaries, (nested) lists, and strings.

        Parameters
        ----------
        dict_data : dict or list or str
            The nested dictionary or list where replacements will be made.

        Returns
        -------
        dict or list or str
            A modified version of the input dict_data with absolute paths.

        Notes
        -----
        This function recursively traverses through the nested structure (dictionary
        or list) and performs replacements on strings using '_adjust_path' method.
        """
        if isinstance(dict_data, dict):
            for key, value in dict_data.items():
                dict_data[key] = self._replace_in_nested_dict(value)
            return dict_data
        elif isinstance(dict_data, list):
            return [self._replace_in_nested_dict(item) for item in dict_data]
        elif isinstance(dict_data, str):
            return self._adjust_path(dict_data)
        else:
            return dict_data

    def _load_and_check_setting(
        self,
        setting: str,
        old_settings: dict,
        new_setting: dict| str| None = None,
    ) -> dict| str:
        """Load and retrieve a specific setting from existing settings.

        This function checks if a specified setting exists in the provided
        `old_settings` dictionary. If the setting is found, it is returned.
        Otherwise, the function retrieves the setting from the original source
        of settings (e.g., loading from a default configuration) and returns it.

        Parameters
        ----------
        setting : str
            The name of the setting to retrieve.
        old_settings : dict
            Dictionary containing existing settings.
        new_settings : dict| str| None
            the new value for the setting, if None is supplied the old setting will
            be loaded.
            default = None

        Returns
        -------
        dict| str
            The retrieved setting value. If the setting is a dictionary, it returns
            the dictionary. If the setting is not found and requires loading, it may
            return a string (e.g., file path or identifier).

        Notes
        -----
        This function serves as a utility to safely retrieve a specific setting.
        It first checks the `old_settings` dictionary. If the setting is not found
        in `old_settings`, it falls back to loading the setting from another source.
        """
        if new_setting is None:
            if setting in old_settings.keys():
                return old_settings[setting]
            else:
                return self.load_settings(original_settings=True)[setting]
        else:
            return new_setting

    def reset_settings(self) -> None:
        """Resets the settings to the original parameters file and overwrite
        the user settings.
        """
        self._save_settings(self.load_settings(original_settings=True))

    def _save_settings(self, new_settings: dict) -> None:
        """Save the updated settings to a YAML file.

        This function writes the provided `new_settings` dictionary to a YAML file
        located in the `.climate` directory within the user's home directory.

        Parameters
        ----------
        new_settings : dict
            Dictionary containing the updated settings to be saved.

        Notes
        -----
        The settings are serialized to YAML format and written to the file specified
        by the path `userdir/.climate/parameters.yaml`.
        """
        if not os.path.exists(os.path.join(os.path.expanduser("~"), ".climate")):
            os.mkdir(os.path.join(os.path.expanduser("~"), ".climate"))

        parameter_file = os.path.join(
            os.path.expanduser("~"), ".climate", "parameters.yaml"
        )
        yaml = ruamel.yaml.YAML()

        with open(parameter_file, "w+") as parameters:
            yaml.dump(new_settings, parameters)

    def load_settings(self, original_settings: bool = False) -> dict:
        """Loads the settings from the user directory, if the settings do not exist in this
        directory the original parameters are loaded.

        Parameters
        ----------
        original_settings : bool, optional
            Flag indicating whether the original settings should be loaded
            default = False

        Returns
        -------
        dict
            The settings as a dict
        """
        if (not original_settings) & os.path.isfile(
            os.path.join(os.path.expanduser("~"), ".climate", "parameters.yaml")
        ):
            settings_path = os.path.join(
                os.path.expanduser("~"), ".climate", "parameters.yaml"
            )
        else:
            settings_path = os.path.join(os.path.dirname(__file__), "parameters.yaml")

        yaml = ruamel.yaml.YAML()
        settings = yaml.load(open(settings_path))

        return self._replace_in_nested_dict(settings) # type: ignore
