from src.world.network.base import Network


class Snapshot:
    """
    A snapshot in time of a network.

    Attributes:
        time: (float) current time of the network
        locations: (list of tuples) locations of all cells in the network
        states: (list of str) state of all cells in the network
        infected_times: (list float) infection times of all cells in the network
    """
    def __init__(self, network: Network):
        self.time = network.time
        self.locations = network.apply_map(lambda c: (c.x, c.y))
        self.states = network.apply_map(lambda c: c.state)
        self.infected_times = [
            None if remove_at_time is None else (remove_at_time - network.INFECTION_TIME)
            for remove_at_time in network.apply_map(lambda c: c.remove_at_time)
        ]
