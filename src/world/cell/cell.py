import uuid

import numpy as np

from src.error import CellError
from src.world.cell.state import State


class Cell:
    """
    Class for a cell in a network.

    Attributes:
        id: (int) unique id for this cell
        x: (int) x coordinate in the world
        y: (int) y coordinate in the world TODO allow just `position`?
        network: (Network) parent networ
        _state: (State) current state (e.g. Susceptible, Infected, Removed)
        _neighbours: (numpy array of Cells) cells connected to this one
        _remove_at_time: (float) time at which this cell should become Removed

    Methods:
        add/remove_neighbour: connect/disconnect another cell
        reset: reset to initial state

    Properties:
        is_safe: (bool) is this node safe

    """

    def __init__(self, x: int, y: int, network: 'Network', state: State = State.S):
        """
        Constructor for the Cell class.

        :param x: (int) position in x coord
        :param y: (int) position in y coord
        :param network: (Network) the network to which this cell belongs
        :param state: (State) starting state of the cell
        """
        self.id = uuid.uuid1().int >> 64
        self.x = x
        self.y = y
        self.network = network
        self._state = state
        self._neighbours = []
        self._remove_at_time = None
        self._n_infected_neighbours = 0
        self._rate_of_getting_infected = 0.0
        self._is_safe = True

    def __repr__(self) -> str:
        return "%s[%d, %d]" % (self.state, self.x, self.y)

    def __str__(self) -> str:
        return "%s[%d, %d]" % (self.state, self.x, self.y)

    def reset(self):
        """
        Reset the node to starting state. Sets all attributes to initial states.

        :return: None
        """
        self._state = State.S
        self._remove_at_time = None
        self._n_infected_neighbours = 0
        self._rate_of_getting_infected = 0.0
        self._is_safe = True

    @property
    def is_safe(self) -> bool:
        """
        Is this cell safe from being infected (no infected neighbours).

        :return: (bool) True iff this cell has a nearly zero rate of infection
        """
        return self._is_safe

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, new_state: State):
        """
        Change the state of the cell.
        Also triggers the network to update any lists.

        :param new_state: (State) new state for this Cell
        :return: None
        """

        if self.state == new_state:
            return

        old_neighbour_rates = [
            cell.rate_of_getting_infected for cell in self._neighbours
            if cell.state == State.S
        ]

        old_rate = self.rate_of_getting_infected
        old_state = self.state
        self._state = new_state

        # TODO this is only for SIR model; usually need to have method for all pairs of old state and new state
        if new_state == State.S:
            pass

        elif new_state == State.I:
            self._remove_at_time = self.network.time + self.network.INFECTION_TIME
            self._rate_of_getting_infected = 0
            for neighbour in self._neighbours:
                neighbour.n_infected_neighbours += 1

        elif new_state == State.R:
            self._remove_at_time = np.inf
            self._rate_of_getting_infected = 0
            for neighbour in self._neighbours:
                neighbour.n_infected_neighbours -= 1

        else:
            raise CellError('Undefined behaviour when state changes to {}'.format(new_state))

        self.network.cell_state_changed(self, old_state, new_state)

        self.network.sum_events_rates = self.network.sum_events_rates + (self.rate_of_getting_infected - old_rate)

        new_neighbour_rates = [
            cell.rate_of_getting_infected for cell in self._neighbours
            if cell.state == State.S
        ]
        difference_rates = sum(new - old for old, new in zip(old_neighbour_rates, new_neighbour_rates))
        self.network.sum_events_rates = self.network.sum_events_rates + difference_rates

    @property
    def remove_at_time(self) -> float:
        """
        Time at which this cell should be removed.

        :return: (float) removal time
        """
        return self._remove_at_time

    @property
    def n_infected_neighbours(self):
        return self._n_infected_neighbours

    @n_infected_neighbours.setter
    def n_infected_neighbours(self, new_value):
        """
        Setting for number of infected neighbours.
        Also updates the rate of infection.

        :param new_value: (int) new number of infected neighbour
        :return: None
        """
        self._n_infected_neighbours = new_value
        if self.state == State.S:
            rate_per_neighbour = self.network.ALPHA + (self.network.BETA * (self._n_infected_neighbours - 1))
            self._rate_of_getting_infected = max(0.0, self._n_infected_neighbours * rate_per_neighbour)
        else:
            self._rate_of_getting_infected = 0.0
        self._is_safe = self._rate_of_getting_infected < 1e-10

    @property
    def rate_of_getting_infected(self):
        """
        Get the infected rate based on the number of infected neighbours.

        :return: None
        """
        return self._rate_of_getting_infected

    def add_neighbour(self, new_cell: 'Cell'):
        """
        Add a new cell as a neighbour of this one.

        :param new_cell: (Cell) the cell to add as a neighbour
        :return: None
        """
        self._neighbours.append(new_cell)

    def remove_neighbour(self, old_cell: 'Cell'):
        """
        Remove one of this cell's neighbours.

        :param old_cell: (Cell) the cell to remove as a neighbour
        :return: None
        """
        self._neighbours.remove(old_cell)

    def neighbour_ids(self):
        """
        Get list of the ID values from all neighbours.

        :return: (list of int) ID values for all neighbours.
        """
        return [n.id for n in self._neighbours]

    def rewire_to_neighbour(self, new_cell: 'Cell', replace: bool = True) -> bool:
        """
        Rewire to a new neighbour, replacing a current one.

        :param new_cell: (Cell) the new cell to add as a neighbour
        :param replace: (bool) replace one of the existing neighbours (randomly selected)
        :return: (bool) True iff replacing succeeded; False if there were no existing neighbours to replace
        """
        if replace:
            if len(self._neighbours) == 0:
                return False
            _replace = np.random.choice(self._neighbours)
            self.remove_neighbour(_replace)
            _replace.remove_neighbour(self)
        self.add_neighbour(new_cell)
        new_cell.add_neighbour(self)
        return True

    @property
    def array_value(self) -> float:
        """
        Create value between [0, 1] depending on the state of the cell.

        :return: (float) in the range [0, 1]
        """
        if self.state == State.S:
            return 0.0
        elif self.state == State.R:
            return 1.0
        else:
            infected_time = self.remove_at_time - self.network.INFECTION_TIME
            return (self.network.time - infected_time) /  self.network.INFECTION_TIME
