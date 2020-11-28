import mlflow


# Find existing runs
from tqdm import tqdm

from src.global_config import GlobalConfig


def delete_unfinished_experiments():
    """
    Find all unfinished experiments (with n_infected != 0) and delete them.

    :return: None
    """
    print('Loading experiments ...')
    mlflow.set_experiment(GlobalConfig().experiment_name)
    df = mlflow.search_runs(experiment_ids=GlobalConfig().experiment_id)
    print('... found {} experiments.'.format(len(df)))

    # Filter to keep only those that were not completed
    df = df[df['metrics.percentage_infected'] != 0.0]

    # Delete the selected experiments
    print('There are {} experiments to be deleted'.format(len(df)))
    for run_id in tqdm(df['run_id']):
        mlflow.delete_run(run_id=run_id)


if __name__ == "__main__":
    delete_unfinished_experiments()
