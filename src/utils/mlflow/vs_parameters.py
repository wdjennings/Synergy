from datetime import timedelta
from typing import Dict, List, Optional, Union

import matplotlib
import numpy as np
from pandas import to_datetime, DataFrame

from src.utils.mlflow.grid import pad_for_parameter_grid
from src.utils.mlflow.load_runs import load_runs

matplotlib.use('TkAgg')


class HeatmapTypes:
    Count = 'Count'
    TimeTaken = 'Time'
    Epidemics1D = 'metrics.is_1d_epidemic'
    Epidemics2D = 'metrics.is_2d_epidemic'
    PercentageRemoved = 'metrics.percentage_removed'

    @staticmethod
    def all():
        return [
            HeatmapTypes.Count,
            HeatmapTypes.TimeTaken,
            HeatmapTypes.Epidemics1D,
            HeatmapTypes.Epidemics2D,
            HeatmapTypes.PercentageRemoved,
        ]


def plot_vs_parameters(parameter_grid: Dict[str, List], type: Union[str, List[str]]= HeatmapTypes.Count,
                       runs_df: Optional[DataFrame] = None, cached: bool = False, **kwargs):
    """
    Plot a heatmap of how many runs for combinations of two parameters.

    :param parameter_grid: (dict) parameters to use as the axes of the heatmap
    :param type: (str) name of heatmap type to show
    :param runs_df: (DataFrame) existing runs table
    :param cached: (bool) use cached experiments (bypass mlflow call)
    :return: None
    """

    parameter_grid = {
        key: values for key, values in parameter_grid.items() if len(values) > 1
    }

    if runs_df is None:
        runs_df = load_runs(cached=cached)

    columns, values = map(list, zip(*parameter_grid.items()))
    heatmap = get_array_values(columns=columns, values=values, type=type, runs_df=runs_df, **kwargs)

    if len(heatmap.shape) == 2:
        plot_heatmap(heatmap, columns, values)

    elif len(heatmap.shape) == 1:
        plot_line(xs=values[0], ys=heatmap, x_label=columns[0], y_label=type)


def plot_line(xs, ys, x_label, y_label):
    import matplotlib.pyplot as plt
    plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel('Mean {}'.format(y_label))
    plt.show()


def plot_heatmap(heatmap, columns, values):
    import matplotlib.pyplot as plt

    xs, ys = values[0], values[1]
    aspect = np.ptp(xs) / np.ptp(ys)
    plt.imshow(heatmap.T, extent=[min(xs), max(xs), min(ys), max(ys)],
               aspect=aspect, origin='lower', cmap='jet', vmin=0)
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.colorbar()
    plt.show()


def get_array_values(columns: List[str], values: List, type: str, runs_df: DataFrame, max_time: float = 20 * 60.0) -> np.ndarray:
    """
    Generate 2d numpy array to be displayed as the heatmap

    :param columns: (list of str) names of parameters to use as the axes of the heatmap
    :param values: (list) parameter values to use as the axes of the heatmap
    :param type: (str) name of heatmap type to show
    :param runs_df: (DataFrame) existing runs table
    :param max_time: (float) cutoff time
    :return: (numpy array), shape `( len(parameter_grid.values()[0]), len(parameter_grid.values()[1]) )`
    """

    if type == HeatmapTypes.Count:
        runs_df = runs_df.groupby(columns).size().reset_index()
        runs_df = runs_df.rename(columns={0: 'Values'})

    elif type == HeatmapTypes.TimeTaken:
        runs_df['Time'] = to_datetime(runs_df['end_time']) - to_datetime(runs_df['start_time'])

        # Any experiment that lasts > 10 mins is probably caused by laptop going to sleep
        outlier_mask = runs_df['Time'] < timedelta(seconds=max_time)

        if sum(~outlier_mask):
            print('Dropping {} rows whose time taken are probably invalid'.format(sum(~outlier_mask)))
            bad_runs = runs_df[~outlier_mask]
            print(bad_runs[[*columns, 'Time']])
            runs_df = runs_df[outlier_mask]

        runs_df = runs_df.groupby(columns).Time.apply(lambda times_taken: np.mean(times_taken).total_seconds())
        runs_df = runs_df.reset_index()
        runs_df = runs_df.rename(columns={'Time': 'Values'})

    elif type in runs_df:
        runs_df = runs_df.groupby(columns)[type].mean().reset_index()
        runs_df = runs_df.rename(columns={type: 'Values'})

    else:
        raise RuntimeError('No known heatmap type {}'.format(type))

    # Convert to filled grid of parameters
    runs_df = pad_for_parameter_grid(runs_df=runs_df, parameter_grid=dict(zip(columns, values)))

    # Convert to array and reshape to 2D
    return runs_df['Values'].values.reshape(tuple(map(len, values)))
