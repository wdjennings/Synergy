from typing import List

from src.world.callbacks.base import Callback


class CallbacksGroup(Callback):
    """
    Class to store a list of Callbacks.

    Attributes:
        _callbacks: (list of Callback) other callbacks to trigger on each event
    """

    def __init__(self, callbacks: List[Callback]):
        """
        Constructor for CallbacksGroup.

        :param callbacks: (list of Callbacks) callbacks to use
        """
        if isinstance(callbacks, list):
            self._callbacks = callbacks
        elif isinstance(callbacks, Callback):
            self._callbacks = [callbacks]
        else:
            self._callbacks = []

    def on_simulation_started(self, network: 'Network'):
        for callback in self._callbacks:
            callback.on_simulation_started(network)

    def should_stop_simulation(self, network: 'Network'):
        return any(callback.should_stop_simulation(network) for callback in self._callbacks)

    def on_simulation_finished(self, network: 'Network'):
        for callback in self._callbacks:
            callback.on_simulation_finished(network)

    def on_event_occurred(self, network: 'Network'):
        for callback in self._callbacks:
            callback.on_event_occurred(network)
