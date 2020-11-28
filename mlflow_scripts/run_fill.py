import time

from tqdm import tqdm

from src.global_config import GlobalConfig
from src.utils.mlflow.fill_out import fill_out_experiment
from src.utils.mlflow.unfinished import ignore_and_delete_unfinished
from src.world.callbacks.mlflow import MLFlowCallback
from src.world.callbacks.snapshot_history import SnapshotHistoryCallback
from src.world.network.rectilinear import Rectilinear2DNetwork

mlflow_callback = MLFlowCallback()
video_callback = SnapshotHistoryCallback()
config = GlobalConfig(filename='config.yml')

while True:

    # Decide what next experiments to run
    new_parameters_df = fill_out_experiment(
        parameter_grid=GlobalConfig().parameter_grid,
        rounding=GlobalConfig().runs_per_batch,
        ignore_rows=ignore_and_delete_unfinished,
        maximum_runs=GlobalConfig().maximum_runs
    )

    if len(new_parameters_df) == 0:
        print('Finished running all experiments.')
        break

    # Sort by size so that we recreate the network fewer times
    new_parameters_df = new_parameters_df.sort_values('metrics.SIZE')
    n_total = sum(new_parameters_df.Count)

    print('\nStarting to run {} simulations\n'.format(n_total))
    start_time = time.time()
    my_simulation = None

    # TODO use multi processing to do several at once
    with tqdm(total=n_total) as progress_bar:
        for index, params in new_parameters_df.iterrows():
            progress_bar.set_description('Simulation {}'.format(params.drop(columns=['Count']).to_dict()))

            alpha, beta = params['metrics.ALPHA'], params['metrics.BETA']
            size, count = int(params['metrics.SIZE']), int(params['Count'])

            # Update network object with parameters (and, if needed, new size)
            if my_simulation is None or size != my_simulation.n_x:
                my_simulation = Rectilinear2DNetwork(shape=(size, size))
            my_simulation.update_disease_type(alpha=alpha, beta=beta)

            # Run several repeats
            for repeat_idx in range(count):
                progress_bar.update(1)
                my_simulation.start_infection_at_center()
                my_simulation.run(callback=mlflow_callback)
                my_simulation.reset()

    print('   ... took {} seconds'.format(time.time() - start_time))
