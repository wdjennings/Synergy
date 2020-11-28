from abc import abstractmethod
from logging import getLogger
from typing import Any, Callable, List, Union, Optional

from src.error import NetworkError
from src.world.callbacks.base import Callback
from src.world.callbacks.group import CallbacksGroup
from src.world.cell import Cell
from src.world.cell.state import State

logger = getLogger(__name__)


class Network:
    """
    Base class for Small World Network simulations.

    Attributes:
        alpha: (float) base synergy-free infection rate
        beta: (float) synergy factor
        tau: (float) duration of infection (sets a normalisation factor for time)
        time: (float) current time of simulation
        extinct: (bool) is the disease extinct now

        _cells: (list of Cells) all cells in the network
        _cells_by_id: (dict of Cells) all cells, keyed by cell.id values
        _cells_by_state: (dict of list of Cells) cells grouped by cell.state values
        _sum_rates: (float) keep track of rates to stop summing

    Properties:
        sum_events_rates: (float) sum of all infection rates (should be sum(_threatened_cells_dict.keys())
        safe_cells: (list of Cells) cells that are in the Susceptible state with zero infection chance
        threatened_cells: (list of Cells) cells that are in the Susceptible state with non-zero infection rate
        infected_cells: (list of Cells) cells that are in the Infected state
        removed_cells: (list of Cells) cells that are in the Removed state
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.0, tau: float = 1.0):
        """
        Constructor for NetworkSimulation.

        :param alpha: (float) base synergy-free infection rate
        :param beta: (float) synergy factor
        :param tau: (float) duration of infection (sets a normalisation factor for time)
        """
        self.ALPHA = alpha
        self.BETA = beta
        self.INFECTION_TIME = tau

        self.time = 0.0
        self.extinct = False
        self._sum_rates = 0.0

        self._cells = self.generate_cells()
        self._cells_by_id = {
            cell.id: cell for cell in self._cells
        }
        self._cells_by_state = {
            state: [cell for cell in self._cells if cell.state == state] for state in State
        }

    def update_disease_type(self, alpha: Optional[float] = None, beta: Optional[float] = None, tau: Optional[float] = None):
        """
        Change the simulation statistics

        :param alpha: (float) NEW base synergy-free infection rate; if None, use previous value
        :param beta: (float) NEW synergy factor; if None, use previous
        :param tau: (float) NEW duration of infection (sets a normalisation factor for time); if None, use previous
        :return:
        """
        self.ALPHA = self.ALPHA if alpha is None else alpha
        self.BETA = self.BETA if beta is None else beta
        self.INFECTION_TIME = self.INFECTION_TIME if tau is None else tau

    def reset(self):
        """
        Reset the simulation to fully Susceptible nodes.
        TODO also reset rewiring?

        :return: None
        """

        self.time = 0
        self.extinct = False
        self._sum_rates = 0.0

        for cell in self._cells:
            cell.reset()

        self._cells_by_state = {
            state: [cell for cell in self._cells if cell.state == state] for state in State
        }

    def run(self, callback: Union[Callback, List[Callback]] = None):
        """
        Run the simulation now, taking snapshots if requested.

        :param callback: (Callback, or list of Callback) callbacks for network during training
        :return: None
        """

        if len(self.infected_cells) == 0:
            raise NetworkError('Unable to run simulation with no infected cells.')

        callback = CallbacksGroup(callback)

        # TODO allow other simulation types (eg fixed time)
        from src.world.simulation.kmc import KineticMonteCarloSimulation
        simulation = KineticMonteCarloSimulation

        callback.on_simulation_started(self)

        while not self.extinct:
            simulation.step_forwards(self)

            callback.on_event_occurred(self)
            if callback.should_stop_simulation(self):
                break

        callback.on_simulation_finished(self)

    def cell_state_changed(self, cell: Cell, old_state: State, new_state: State):
        """
        Called after a cell's state has changed.
        Moves the cell in the self._cells_by_state lists.

        :param cell: (Cell) cell whose state has changed
        :param old_state: (State) old state of cell
        :param new_state: (State) new state of cell
        :return: None
        """

        if old_state == new_state:
            return

        if cell in self._cells_by_state[old_state]:
            self._cells_by_state[old_state].remove(cell)
        else:
            logger.error(
                'Cell {} changed state from {} to {}, '.format(cell, old_state, new_state) +
                'but did not exist in _cells_by_state[{}]'.format(old_state)
            )

        if cell not in self._cells_by_state[new_state]:
            self._cells_by_state[new_state].append(cell)
        else:
            logger.error(
                'Cell {} changed state from {} to {}, '.format(cell, old_state, new_state) +
                'but already existed in _cells_by_state[{}]'.format(new_state)
            )

    @abstractmethod
    def generate_cells(self, *args, **kwargs) -> List[Cell]:
        """
        Generate a list of the cells to represent the map.

        :param args, kwargs: options for creating the map (eg size)
        :return: (list of Cell) all cells in the map
        """
        pass

    @abstractmethod
    def is_epidemic(self, epidemic_type: str) -> bool:
        """
        Determine if network is currently in an epidemic of the given type.

        :param epidemic_type: (str) epidemic type (e.g. 1d, 2d, ...)
        :return: (bool) true iff the network is currently in an epidemic of the given type.
        """
        raise NotImplementedError('Subclass and implement')

    @abstractmethod
    def epidemic_type(self) -> str:
        """
        Get the epidemic type of the network.

        :return: (str) type of epidemic
        """
        pass

    @property
    def safe_cells(self) -> List[Cell]:
        """
        Cells which are at no risk of being infected, i.e. have no infected neighbours.

        :return: (list of Cell) safe cells
        """
        return [cell for cell in self._cells_by_state[State.S] if cell.is_safe]

    @property
    def threatened_cells(self) -> List[Cell]:
        """
        Cells which are at risk of being infected, i.e. have non-zero number of infected neighbours.

        :return: (list of Cell) threatened cells
        """
        return [cell for cell in self._cells_by_state[State.S] if not cell.is_safe]

    @property
    def infected_cells(self) -> List[Cell]:
        """
        Cells which are currently infected.

        :return: (list of Cell) infected cells
        """
        return self._cells_by_state[State.I]

    @property
    def removed_cells(self) -> List[Cell]:
        """
        Cells which are removed.

        :return: (list of Cell) removed cells
        """
        return self._cells_by_state[State.R]

    @property
    def percentage_safe(self) -> float:
        """
        What percentage of the cells are currently safe.

        :return:
        """
        return 100.0 * float(len(self.safe_cells)) / float(len(self._cells))

    @property
    def percentage_infected(self) -> float:
        """
        What percentage of the cells are currently infected.

        :return:
        """
        return 100.0 * float(len(self.infected_cells)) / float(len(self._cells))

    @property
    def percentage_removed(self) -> float:
        """
        What percentage of the cells are currently removed.

        :return:
        """
        return 100.0 * float(len(self.removed_cells)) / float(len(self._cells))

    @property
    def next_remove_time(self) -> float:
        """
        Time at which the next cell will transition to Removed state.

        :return: (float) time for next Remove event
        """
        return min([cell.remove_at_time for cell in self.infected_cells])

    @property
    def sum_events_rates(self) -> float:
        """
        Sum of all infection events.

        :return: (float) sum of all infection events
        """
        return self._sum_rates

    @sum_events_rates.setter
    def sum_events_rates(self, new_rates: float):
        """
        Setter for the sum of all events rates.

        :param new_rates: (float) new events rates
        :return: None
        """
        self._sum_rates = new_rates

    def apply_map(self, function: Callable[[Cell], Any]) -> List[Any]:
        """
        Map a function onto each Cell in the network, and return as list.

        :param function: (callable) convert a Cell into some other value
        :return: (list of Any) one item for each Cell
        """
        return list(map(function, self._cells))
