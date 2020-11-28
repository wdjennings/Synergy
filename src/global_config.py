from typing import Dict, Union, List

import mlflow
import numpy as np
import yaml

from src.utils.singleton import Singleton


class GlobalConfigError(BaseException):
    pass


class GlobalConfig(metaclass=Singleton):
    """
    Class to handle global config details.
    """
    def __init__(self, filename: str = 'config.yml'):
        config_dict = self.load_config(filename)
        self.parameter_grid = GlobalConfig.parse_param_grid(config_dict)
        self.experiment_name = config_dict.pop('experiment_name')
        self.maximum_runs = config_dict.pop('maximum_runs')
        self.runs_per_batch = config_dict.pop('runs_per_batch')

        self.check_unused_keys(config_dict)

        mlflow.set_experiment(self.experiment_name)
        self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = self.experiment._experiment_id

    @staticmethod
    def load_config(config_file: str) -> Dict:
        with open(config_file, 'r') as stream:
            return yaml.load(stream)

    @staticmethod
    def parse_param_grid(config_dict):
        parameter_grid = config_dict.pop('parameter_grid')
        return {
            key: GlobalConfig.expand_parameter_grid_axis(value) for key, value in parameter_grid.items()
        }

    @staticmethod
    def expand_parameter_grid_axis(values: Union[str, List]) -> List:
        """
        Convert config string to list of values for parameter axis.

        :param values:
        :return:
        """

        if isinstance(values, list):
            return values

        if isinstance(values, str):

            if values.startswith('arange'):
                parts_str = values.replace('arange', '')[1:-1]
                min, max, step = map(float, parts_str.split(', '))
                return [np.round(value, decimals=2) for value in np.arange(min, max, step)]

        raise GlobalConfigError(
            'Unable to parse parameter axis values: {}'.format(values)
        )

    @staticmethod
    def check_unused_keys(config_dict):
        if len(config_dict) > 0:
            raise GlobalConfigError(
                'Config file contains unhandled keys: {}'.format(list(config_dict.keys()))
            )
