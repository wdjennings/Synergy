from typing import List, Callable

from pandas import DataFrame

from src.utils.mlflow.load_runs import load_runs


def parameter_run_counts(parameter_names: List[str], ignore_rows: Callable[[DataFrame], DataFrame] = None,
                         runs_df: DataFrame = None) -> DataFrame:
    """
    Get counts of existing experiments, for the given parameter names.

    :param parameter_names: (list of str) names of parameters to group by
    :param ignore_rows: (callable) ignore some experiments
    :param runs_df: (DataFrame) existing experiment results
    :return: (DataFrame) counts of all existing experiments
    """

    if runs_df is None:
        runs_df = load_runs()

    columns = [*parameter_names, 'run_id']

    if len(runs_df) == 0:
        return DataFrame(columns=[*columns, 'Count'])

    missing = [column for column in columns if column not in runs_df]
    if missing:
        raise KeyError(
            'Could not get counts for existing experiments with parameters \'{}\'.'.format(columns) +
            ' The following columns were missing: {}'.format(missing)
        )

    if ignore_rows:
        runs_df = ignore_rows(runs_df)

    runs_df = runs_df.groupby(parameter_names).count()
    runs_df = runs_df.reset_index()[columns].rename(columns={'run_id': 'Count'})
    return runs_df
