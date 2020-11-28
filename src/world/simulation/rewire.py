import numpy as np

from src.world.network.base import Network


def rewire_network(network: Network, probability: float):
    """
    Rewire the network with some random chance for each connection.

    :param network: (NetworkSimulation) the network
    :param probability: (float) probability of any individual node getting rewired
    :return: None
    """

    n_cells = len(network._cells)

    # TODO handle this better
    n_cells_to_rewire = np.random.binomial(n_cells, probability)
    rewire_pair_idxs = np.random.choice(range(n_cells), size=(n_cells_to_rewire, 2))
    failed_rewire_pairs = np.where(rewire_pair_idxs[:, 0] == rewire_pair_idxs[:, 1])[0]

    if len(failed_rewire_pairs) > 0:
        print('WARNING: %d node rewired to self' % (len(failed_rewire_pairs)))

    for from_index, to_index in rewire_pair_idxs:
        _from = network._cells[from_index]
        _to = network._cells[to_index]
        success = _from.rewire_to_neighbour(_to)

        if not success:
            print('WARNING: failed rewiring due to no neighbours')
