from typing import Union
import ruamel.yaml
import os

class alignmentCalculatorConfig():

    def __init__(self):
        pass

    def config(
        self,
        main_pacta_file: dict = None,
        company_information_file: str = None,
        economic_weights: dict = None,
        production_thresholds: dict = None,
        scenario_data: dict = None,
        sectoral_approach: dict = None,
        save_changes: bool = True,
    ) -> Union[None, dict]:
        """Configure the settings for the alignment calculator.

        The adjusted settings will be saved in the .pacta folder in the userdir.
        The settings are in the form of a YAML file and can also be edited manually.

        Parameters
        ----------
        main_pacta_file : dict, optional
            Dictionary containing main pacta file settings.
            The dict should adhere to the following format
            {year: [file_location_company_analytics, file_location_company_indicators],
            ....}
            default= None
        company_information_file : str, optional
            Absolute path to the company information file as a string
            default= None
        economic_weights : dict, optional
            Dictionary containing economic weights, containing a weight for each sector
            {'power':800, ...}
            default= None
        production_thresholds : dict, optional
            Dictionary containing production thresholds, containing both a
            asset_ratio and a turnover_ratio for each sector
            {'power':{'asset_ratio':0.0003, 'turnover_ratio':0.001}, ...}
            default= None
        scenario_data: dict, optional
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
        sectoral_approach : dict, optional
            Dictionary containing sectoral approach settings.
            default= None

        Notes
        -----
        This function loads existing settings, updates them with provided new settings,
        and then saves the updated settings. If a parameter is None, the corresponding
        setting is not updated and remains unchanged. All paths provided in the settings
        should be absolute paths.
        """
        old_settings = self.load_settings()
        new_settings = dict()

        new_settings["main_pacta_file"] = (
            main_pacta_file
            if main_pacta_file is not None
            else self._load_and_check_setting("main_pacta_file", old_settings)
        )
        new_settings["company_information_file"] = (
            company_information_file
            if company_information_file is not None
            else self._load_and_check_setting("company_information_file", old_settings)
        )
        new_settings["economic_weights"] = (
            economic_weights
            if economic_weights is not None
            else self._load_and_check_setting("economic_weights", old_settings)
        )
        new_settings["production_thresholds"] = (
            production_thresholds
            if production_thresholds is not None
            else self._load_and_check_setting("production_thresholds", old_settings)
        )
        new_settings["scenario_data"] = (
            scenario_data
            if scenario_data is not None
            else self._load_and_check_setting("scenario_data", old_settings)
        )
        new_settings["sectoral_approach"] = (
            sectoral_approach
            if sectoral_approach is not None
            else self._load_and_check_setting("sectoral_approach", old_settings)
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
        self,
        dict_data: Union[dict, str, list]
    ) -> Union[dict, str, list]:
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
            old_settings: dict) -> Union[dict, str]:
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

        Returns
        -------
        Union[dict, str]
            The retrieved setting value. If the setting is a dictionary, it returns
            the dictionary. If the setting is not found and requires loading, it may
            return a string (e.g., file path or identifier).

        Notes
        -----
        This function serves as a utility to safely retrieve a specific setting.
        It first checks the `old_settings` dictionary. If the setting is not found
        in `old_settings`, it falls back to loading the setting from another source.
        """
        if setting in old_settings.keys():
            return old_settings[setting]
        else:
            return self.load_settings(original_settings=True)[setting]


    def reset_settings(
            self) -> None:
        """Resets the settings to the original parameters file and overwrite
        the user settings.
        """
        self._save_settings(self.load_settings(original_settings=True))


    def _save_settings(
            self,
            new_settings: dict) -> None:
        """Save the updated settings to a YAML file.

        This function writes the provided `new_settings` dictionary to a YAML file
        located in the `.pacta` directory within the user's home directory.

        Parameters
        ----------
        new_settings : dict
            Dictionary containing the updated settings to be saved.

        Notes
        -----
        The settings are serialized to YAML format and written to the file specified
        by the path `userdir/.pacta/parameters.yaml`.
        """
        if not os.path.exists(os.path.join(os.path.expanduser("~"), ".pacta")):
            os.mkdir(os.path.join(os.path.expanduser("~"), ".pacta"))

        parameter_file = os.path.join(os.path.expanduser("~"), ".pacta", "parameters.yaml")
        yaml = ruamel.yaml.YAML()

        with open(parameter_file, "w+") as parameters:
            yaml.dump(new_settings, parameters)


    def load_settings(
            self,
            original_settings: bool = False) -> dict:
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
            os.path.join(os.path.expanduser("~"), ".pacta", "parameters.yaml")
        ):
            settings_path = os.path.join(
                os.path.expanduser("~"), ".pacta", "parameters.yaml"
            )
        else:
            settings_path = os.path.join(os.path.dirname(__file__), "parameters.yaml")

        yaml = ruamel.yaml.YAML()
        settings = yaml.load(open(settings_path))

        return self._replace_in_nested_dict(settings)
