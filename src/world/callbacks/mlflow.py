import os
import tempfile
from logging import getLogger

import imageio
import mlflow
import numpy as np
from mlflow import log_metric, start_run, end_run

from src.global_config import GlobalConfig
from src.world.callbacks.base import Callback
from src.world.network.rectilinear import EpidemicType2D

logger = getLogger(__name__)


class MLFlowCallback(Callback):
    """
    Callback to handle storing results in mlflow.

    Attributes:
        results_dir: (str) directory in which to store results
        log_histories: (bool) also store the history of
        save_image: (bool) also save image of final state
    """

    def __init__(self, results_dir: str = 'results', log_histories: bool = False, save_images: bool = False):
        self.results_dir = results_dir
        self.log_histories = log_histories
        self.save_images = save_images
        mlflow.set_experiment(GlobalConfig().experiment_name)

    def on_simulation_started(self, network: 'Network'):
        start_run()

        # Choose, set, and store a random seed
        random_seed = int(np.random.randint(0, int(2**32) - 1))
        np.random.seed(random_seed)
        log_metric("NUMPY_RANDOM_STATE", random_seed)

        log_metric("ALPHA", network.ALPHA)
        log_metric("BETA", network.BETA)
        log_metric("INFECTION_TIME", network.INFECTION_TIME)
        log_metric("SIZE", network.n_x)

    def on_event_occurred(self, network: 'Network'):
        if self.log_histories:
            log_metric("time", network.time)
            log_metric("percentage_safe", network.percentage_safe)
            log_metric("percentage_infected", network.percentage_infected)
            log_metric("percentage_removed", network.percentage_removed)

    def on_simulation_finished(self, network: 'Network'):
        log_metric("time", network.time)
        log_metric("percentage_safe", network.percentage_safe)
        log_metric("percentage_infected", network.percentage_infected)
        log_metric("percentage_removed", network.percentage_removed)

        log_metric("is_1d_epidemic", float(network.is_epidemic(EpidemicType2D.OneDimensionalType)))
        log_metric("is_2d_epidemic", float(network.is_epidemic(EpidemicType2D.TwoDimensionalType)))

        with tempfile.TemporaryDirectory() as tmpdirname:
            values = network.to_array()

            np.save(os.path.join(tmpdirname, 'final_state.npy'), values)
            np.save(os.path.join(tmpdirname, 'final_shader_x.npy'), network._shader_x.reshape((1, -1)))
            np.save(os.path.join(tmpdirname, 'final_shader_y.npy'), network._shader_y.reshape((1, -1)))

            if self.save_images:
                save_array_as_image(os.path.join(tmpdirname, 'final_state.png'), values)
                save_array_as_image(os.path.join(tmpdirname, 'final_shader_x.png'), network._shader_x.reshape((1, -1)))
                save_array_as_image(os.path.join(tmpdirname, 'final_shader_y.png'), network._shader_y.reshape((1, -1)))

            mlflow.log_artifacts(tmpdirname)

        end_run()


def save_array_as_image(filename, array):
    imageio.imwrite(filename, 255 * array.astype(np.uint8))
