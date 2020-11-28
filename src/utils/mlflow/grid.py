import itertools
from typing import Dict, List

from pandas import DataFrame


def pad_for_parameter_grid(runs_df: DataFrame, parameter_grid: Dict[str, List]) -> DataFrame:
    """
    Add rows to the DataFrame, so that each parameter combination has exactly one row.

    :param runs_df: (DataFrame) existing experiment runs
    :param parameter_grid: (dict) parameter grid
    :return: (DataFrame) new DataFrame with extra rows added for missing parameter combinations
    """
    columns, values = map(list, zip(*parameter_grid.items()))
    grid_df = runs_df.set_index(columns)

    # Handle 1D case differently
    if len(values) == 1:
        new_index_values = values[0]
    else:
        new_index_values = list(itertools.product(*values))

    grid_df = grid_df.reindex(new_index_values)
    return grid_df.reset_index()
