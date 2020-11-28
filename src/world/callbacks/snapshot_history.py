from copy import deepcopy
from typing import List, Iterable, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.world.callbacks.base import Callback
from src.world.cell import State
from src.world.snapshot import Snapshot

s_color = [0.325, 0.718, 0.306, 1.0]
i_color = [1.0, 0.0, 0.0, 1.0]
r_color = [0.0, 0.0, 0.0, 1.0]


class SnapshotHistoryCallback(Callback):
    """
    Callback to catch and store snapshot objects at each time step.

    Attributes:
        dt_snapshots: (float, or None) how often to take snapshots
        _history: (list of Snapshot) snapshot histories
    """

    def __init__(self, dt_snapshots: Optional[float] = 1.0):
        """
        Constructor for SnapshotHistoryCallback.

        :param dt_snapshots: (float, or None) how often to take snapshots
        """
        self.dt_snapshots = dt_snapshots
        self._history = []

    def reset(self):
        """ Reset callback to initial state """
        self._history = []

    @property
    def last_snapshot(self):
        return self._history[-1] if self._history else None

    def on_event_occurred(self, network: 'Network'):
        self._history.extend(
            recent_snapshots(
                network=network,
                last_snapshot=self.last_snapshot,
                dt_snapshots=self.dt_snapshots
            )
        )

    def on_simulation_finished(self, network: 'Network'):
        pass

    def on_simulation_started(self, network: 'Network'):
        """
        Simulation started -- store the initial parameters, so the video can be made at the end.
        :param network:
        :return:
        """
        self.n_x = network.n_x
        self.n_y = network.n_y
        self.ALPHA = network.ALPHA
        self.BETA = network.BETA
        self.INFECTION_TIME = network.INFECTION_TIME
        self.reset()

    def make_video(self, filename: str = "simulation.mp4", fps: int = 5, marker: str = 's', progress_bar: bool = False):
        """
        Generate a video from a network's snapshot history.

        :param filename: (str) name to store video file
        :param fps: (int) frames per second
        :param marker: (str) mpl marker type
        :param progress_bar: (bool) show progressbar while making frames
        :return: None
        """

        figsize = (8, 8)
        marker_width = 600.0 / self.n_x

        import os
        if os.path.exists(filename):
            os.remove(filename)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Simulation')
        ax.set_xlim(-0.5, self.n_x-0.5)
        ax.set_ylim(-0.5, self.n_y-0.5)

        # im = ax.imshow(rand(300, 300), cmap='gray', interpolation='nearest')
        # im.set_clim([0, 1])
        # fig.set_size_inches([5, 5])

        snapshot = self._history[0]
        xs, ys = zip(*snapshot.locations)
        states = np.array(self._history[0].states)
        colors = np.array([s_color]*len(states))
        scatter = plt.scatter(xs, ys, facecolors=colors, s=marker_width*marker_width, marker=marker)

        plt.tight_layout()

        bar = tqdm(total=len(self._history)) if progress_bar else None

        def update_scatter(n: int):
            # TODO should be handled inside the network class visualisor?
            if progress_bar:
                bar.set_description('Making frame {} of {}'.format(n + 1, len(self._history)))
                bar.update()
            snapshot = self._history[n]
            states = np.array(snapshot.states)
            colors = np.array([s_color]*len(states))

            colors[states == State.R, :] = r_color

            # Fractional red color based on how long until it dies
            if np.sum(states == State.I) > 0:
                time = snapshot.time
                infected_times = snapshot.infected_times
                infected_times = np.array(infected_times)[states == State.I]
                ages = time - np.array(infected_times)
                fraction_ages = 1.0 - (ages / self.INFECTION_TIME)
                age_colours = np.array([i_color] * len(infected_times))
                age_colours[:, :2] = np.multiply(age_colours[:, :2].T, fraction_ages).T
                colors[states == State.I, :] = np.clip(age_colours, 0, 1)

            scatter.set_facecolor(colors)

            ax.set_title("Time = %.2f" % snapshot.time)

            return scatter

        ani = animation.FuncAnimation(fig, update_scatter, len(self._history), interval=100.0 / fps)
        writer = animation.writers['ffmpeg'](fps=30)
        ani.save(filename, writer=writer, dpi=100)

        if bar:
            bar.set_description('Video saved to \'{}\''.format(filename))
            bar.close()


def recent_snapshots(network: 'Network', last_snapshot: Snapshot, dt_snapshots: float) -> List[Snapshot]:
    """
    Get the snapshots since the last snapshot time.

    :param network: (Network) network to get snapshots for
    :param last_snapshot: (float) last snapshot taken
    :param dt_snapshots: (float) time step between each snapsot; if None, take snapshot at each (uneven) time step
    :return: (list of Snapshot) all snapshots needed since last snapshot time
    """

    # Snapshot at every step
    if dt_snapshots is None or last_snapshot is None:
        return [Snapshot(network)]

    # Disease is dead -- take one final snapshot
    elif network.extinct:
        snapshot = Snapshot(network)
        snapshot.time = last_snapshot.time + dt_snapshots
        return [snapshot]

    # Time has advanced enough to take snapshot(s)
    elif network.time > last_snapshot.time + dt_snapshots:
        return list(snapshots_at_intervals(network, last_snapshot, dt_snapshots))

    else:
        return []


def snapshots_at_intervals(network: 'Network', last_snapshot: Snapshot, dt_snapshots: float) \
        -> Iterable[Snapshot]:
    """
    Get the list of snapshots since the last snapshot time.

    :param network: (Network) network to get snapshots for
    :param last_snapshot: (Snapshot) last snapshot taken
    :param dt_snapshots: (float) time step between each snapsot; if None, take snapshot at each (uneven) time step
    :return: (generator of Snapshots)
    """
    snapshot = Snapshot(network)
    snap_time = last_snapshot.time
    while snap_time + dt_snapshots < network.time:
        snap_time += dt_snapshots
        snapshot = deepcopy(snapshot)
        snapshot.time = snap_time
        yield snapshot
