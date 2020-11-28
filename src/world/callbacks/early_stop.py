from src.world.callbacks.base import Callback


class EarlyStoppingCallback(Callback):
    """
    Class to handle early stopping once an Epidemic is reached.

    Attributes:
        epidemic_type: (str) type of epidemic at which to stop
    """

    def __init__(self, epidemic_type: str):
        """
        Constructor for EarlyStoppingCallback.

        :param epidemic_type: (str) type of epidemic at which to stop
        """
        self._epidemic_type = epidemic_type

    def should_stop_simulation(self, network: 'Network') -> bool:
        """
        Should the simulation stop running now.

        :param network: (Network) network to get data from
        :return: (bool) should the simulation stop
        """
        return network.is_epidemic(self._epidemic_type)
