import os

import mlflow
from pandas import DataFrame, read_csv

from src.global_config import GlobalConfig


def load_runs(cached: bool = False) -> DataFrame:
    """
    Load experiment runs, using caching for faster loading while testing.

    :param cached: (bool) if True, use caching
    :return: (DataFrame) experiment runs
    """

    if cached and os.path.exists('existing_df.csv'):
        print('WARNING: getting CACHED existing experiments results ...')
        runs_df = read_csv('existing_df.csv')
        print('... found {} existing experiments.'.format(len(runs_df)))
        return runs_df

    else:
        print('Getting existing experiments results ...')
        runs_df = mlflow.search_runs(experiment_ids=GlobalConfig().experiment_id)
        print('... found {} existing experiments.'.format(len(runs_df)))

        if cached:
            print('WARNING: saving existing experiments results to cached file')
            runs_df.to_csv('existing_df.csv', index=False)

        return runs_df
