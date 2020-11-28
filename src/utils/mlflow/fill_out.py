import itertools
from typing import List, Dict, Callable, Optional

from pandas import DataFrame

from src.utils.mlflow.count import parameter_run_counts


def fill_out_experiment(parameter_grid: Dict[str, List], rounding: int = 10, maximum_runs: Optional[int] = None,
                        ignore_rows: Callable[[DataFrame], DataFrame] = None) -> DataFrame:
    """
    Create a DataFrame to fill out the current experiment data set.

    :param parameter_grid: (dict) parameter values to search for
    :param rounding: (int) round up to nearest multiple of this
    :param maximum_runs: (int) maximum number of experiments to run for each combination
    :param ignore_rows: (callable) ignore some experiments
    :return: (DataFrame) experiments to be run
    """
    columns, values = map(list, zip(*parameter_grid.items()))
    run_counts_df = parameter_run_counts(parameter_names=columns, ignore_rows=ignore_rows)

    desired_df = DataFrame(data=itertools.product(*values), columns=columns)
    required_df = required_rows(run_counts_df, desired_df, columns, rounding, maximum_runs)

    return required_df


def next_run_count(existing_df: DataFrame, rounding: int = 10) -> int:
    """
    How many runs to use next.

    :param existing_df: (DataFrame) existing experiments that have already been run
    :param rounding: (int) round up to nearest multiple of this
    :return: (int) how many experiments to fill out to now
    """
    smallest_count = existing_df['Count'].min()
    if smallest_count > 0:
        return ((smallest_count // rounding) * rounding) + rounding
    else:
        return rounding


def required_rows(existing_df, desired_df, columns: List[str], rounding: int = 10, maximum_runs: int = None) -> DataFrame:
    """
    Decide how many of each experiment to run, based on the number of existing and desired experiments.

    :param existing_df: (DataFrame) existing experiments that have already been run
    :param desired_df: (DataFrame) new experiments that are requested
    :param columns: (list of str) which columns to use for deciding existing experiments
    :return: (DataFrame) new experiments to run, with Count column saying how many of each set to run
    """

    if len(existing_df) == 0:
        desired_df['Count'] = rounding
        return desired_df

    desired_df, existing_df = desired_df.set_index(columns), existing_df.set_index(columns)

    # Drop undesired rows before deciding the count (otherwise is always too low)
    keep_indices = [index for index in desired_df.index if index in existing_df.index]
    existing_df = existing_df.loc[keep_indices]

    # Set desired counts
    next_count = next_run_count(existing_df, rounding)
    print('Will use {} runs next'.format(next_count))
    desired_df['Count'] = next_count

    if maximum_runs is not None:
        desired_df[desired_df['Count'] > maximum_runs] = maximum_runs

    # Figure out the difference between desired and existing counts
    required_df = desired_df - existing_df

    # Fill missing experiments that have not been started yet
    required_df[required_df.isna()] = desired_df[required_df.isna()]

    # Drop undesired rows (TODO no longer needed?)
    required_df = required_df.loc[desired_df.index]

    # Drop zero (or negative) rows
    required_df = required_df.astype(int).reset_index()
    required_df = required_df[required_df.Count > 0]

    return required_df

