import mlflow
from pandas import DataFrame


def ignore_and_delete_unfinished(df: DataFrame) -> DataFrame:
    """
     Find and delete 'unfinished' experiments, returning the clean df of finished experiments.

     :param df: (DataFrame) all existing experiments
     :return: (DataFrame) only finished experiments
     """

    delete_mask = df['metrics.percentage_infected'] != 0.0

    delete_df = df[delete_mask]
    if len(delete_df) > 0:
        print('There are {} experiments to be deleted'.format(len(delete_df)))
        for run_id in delete_df['run_id']:
            mlflow.delete_run(run_id=run_id)

    return df[~delete_mask]


def ignore_unfinished(df: DataFrame) -> DataFrame:
    """
     Filter out any 'unfinished' experiments, returning the clean df of finished experiments.

     :param df: (DataFrame) all existing experiments
     :return: (DataFrame) only finished experiments
     """
    return df[df['metrics.percentage_infected'] == 0.0]
