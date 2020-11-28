import itertools
import time
from argparse import ArgumentParser

import mlflow
from tqdm import tqdm

from src.global_config import GlobalConfig
from src.utils.argparser.linspacer import linspacer
from src.world.callbacks.mlflow import MLFlowCallback
from src.world.network.rectilinear import Rectilinear2DNetwork

parser = ArgumentParser(description="Choose some simulation parameters.")
parser.add_argument('-size', metavar='N', type=int, help='Size of the square network', default=21)
parser.add_argument('-alphas', nargs='+', help='Alpha values (min, max, [count])', **linspacer(0.2, 2.0, 20))
parser.add_argument('-betas', nargs='+', help='Beta values (min, max, [count])', **linspacer(0, 0, 1))
args = parser.parse_args()

mlflow_callback = MLFlowCallback()
mlflow.set_experiment(GlobalConfig().experiment_name)


my_simulation = Rectilinear2DNetwork(shape=(args.size, args.size))

print('Running models for network size {}'.format(args.size))
start_time = time.time()

for alpha, beta in tqdm(list(itertools.product(args.alphas, args.betas))):
    my_simulation.update_disease_type(alpha=alpha, beta=beta)
    my_simulation.start_infection_at_center()
    my_simulation.run(callback=mlflow_callback)
    my_simulation.reset()

print('   ... took {} seconds'.format(time.time() - start_time))
