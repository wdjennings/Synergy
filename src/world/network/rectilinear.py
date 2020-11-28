from typing import List, Tuple

from numpy import zeros, array, ndarray

from src.world.cell import Cell, State
from src.world.network.base import Network


class EpidemicType2D:
    NoType = 'None'
    OneDimensionalType = '1D'
    TwoDimensionalType = '2D'


class Rectilinear2DNetwork(Network):
    """
    2D Network with rectilinear (grid-like) connections.

    Attributes:
        n_x: (int) size of the map in x direction
        n_y: (int) size of the map in y direction
        _cell_map_2d: (2d numpy array) map of cells
        _shader_x: (1d numpy array) boolean array measuring whether each column (x-index) has ANY infected cells in it
        _shader_y: (1d numpy array) boolean array measuring whether each row (y-index) has ANY infected cells in it
    """
    adjacent_cell_vectors = [
        (-1, 0),   # West
        (1, 0),    # East
        (0, 1),    # North
        (0, -1)    # South
    ]

    def __init__(self, shape: Tuple[int, int], alpha: float = 0.7, beta: float = 0.0, tau: float = 1.0):
        """
        Constructor for RectilinearNetwork.

        :param shape: (tuple of int) size of the map in direction
        """
        n_x, n_y = shape
        self.n_x = n_x
        self.n_y = n_y
        self._shader_x = zeros(self.n_x, dtype=bool)
        self._shader_y = zeros(self.n_y, dtype=bool)
        self._cell_map_2d = None
        super().__init__(alpha=alpha, beta=beta, tau=tau)

    def reset(self):
        super().reset()
        self._shader_x[:] = False
        self._shader_y[:] = False

    def to_array(self) -> ndarray:
        """
        Convert to array for the different cell states.

        :return: (numpy array) 2D array, with values 0->1
        """
        return array([
            [cell.array_value for cell in row] for row in self._cell_map_2d
        ])

    def generate_cells(self) -> List[Cell]:
        """
        Generate a list of the cells to represent the map.

        :return: (list of Cell) all cells in the map
        """

        self._cell_map_2d = array([
            [Cell(x, y, network=self) for y in range(self.n_y)]
            for x in range(self.n_x)
        ])

        for column in self._cell_map_2d:
            for cell in column:
                for (other_x, other_y) in self.adjacent_cell_coords(cell.x, cell.y):
                    self._cell_map_2d[other_x][other_y].add_neighbour(cell)

        return list(self._cell_map_2d.flatten())

    def adjacent_cell_coords(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get list of adjacent cells to a specific one.

        :param x: (int) x location
        :param y: (int) y location
        :return: (list of tuples) all locations that are adjacent to (x, y)
        """
        return [self.wrap_periodic(x + dx, y + dy) for (dx, dy) in self.adjacent_cell_vectors]

    def wrap_periodic(self, x: int, y: int ) -> Tuple[int, int]:
        """
        Wrap a location using Periodic Boundary Conditions.

        :param x: (int) x location
        :param y: (int) y location
        :return: (tuple) x and y indications that are corrected to be within bounds.
        """
        return x % self.n_x, y % self.n_y

    def is_epidemic(self, type: str) -> bool:
        """
        Test whether this network is currently in a specific epidemic state.

        :param type: (str) type of epidemic
        :return: (bool) True iff network is in the specified epidemic state.
        """
        if type == EpidemicType2D.OneDimensionalType:
            return all(self._shader_x) != all(self._shader_y)

        elif type == EpidemicType2D.TwoDimensionalType:
            return all(self._shader_x) and all(self._shader_y)

        else:
            raise NotImplementedError('Unknown Epidemic type: %s' % type)

    def epidemic_type(self):
        """
        Determine the maximal epidemic type (i.e. 2D, otherwise 1D, otherwise none)
        :return: (str) the type of epidemic
        """
        epidemic_x = all(self._shader_x)
        epidemic_y = all(self._shader_y)
        if epidemic_x != epidemic_y:
            return EpidemicType2D.OneDimensionalType
        elif epidemic_x and epidemic_y:
            return EpidemicType2D.TwoDimensionalType
        else:
            return EpidemicType2D.NoType

    def start_infection_at_center(self):
        """ Begin the infection at the central cell """
        cx, cy = int(self.n_x / 2), int(self.n_y / 2)
        cell_to_infect = self._cell_map_2d[cx, cy]
        cell_to_infect.state = State.I

    def cell_state_changed(self, cell: Cell, old_state: State, new_state: State):
        """ Wrap superclass method, to track shaders each time a cell is infected. """
        if new_state == State.I:
            self._shader_x[cell.x] = True
            self._shader_y[cell.y] = True
        super().cell_state_changed(cell, old_state, new_state)
