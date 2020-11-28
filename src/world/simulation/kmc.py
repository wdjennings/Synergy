from logging import getLogger

import numpy as np

from src.world.cell import State
from src.world.network.base import Network
from src.world.simulation.base import Simulation

logger = getLogger(__name__)


class KineticMonteCarloSimulation(Simulation):

    @staticmethod
    def step_forwards(network: Network):
        """
        Take a time step until the next random- or fixed-time event.
        TODO only for SIR model?

        :param network: (Network) network to update
        :return: None
        """

        if len(network.infected_cells) == 0:
            network.extinct = True
            return

        # Sample a random time for the next step to happen
        if network.sum_events_rates > 1e-10:
            random_u = np.random.uniform()
            delta_t = (1.0 / network.sum_events_rates) * np.log(1.0 / random_u)
        else:
            delta_t = np.inf

        # Remove event occurs before infection event?
        if network.time + delta_t >= network.next_remove_time:
            network.time = network.next_remove_time
            cell_to_remove = network.infected_cells[0]
            cell_to_remove.state = State.R

        # Infection event occurs
        else:
            network.time = network.time + delta_t
            threatened_rates = [cell.rate_of_getting_infected for cell in network.threatened_cells]
            normalised_rates = np.array(threatened_rates) / network.sum_events_rates
            cell_to_infect = np.random.choice(network.threatened_cells, p=normalised_rates)
            cell_to_infect.state = State.I
