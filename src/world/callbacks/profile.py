import cProfile

from src.world.callbacks.base import Callback


class ProfilingCallback(Callback):

    def __init__(self):
        self._pr = cProfile.Profile()
        self._pr.enable()

    def print_stats(self):
        self._pr.print_stats(sort="calls")

    def disable(self):
        self._pr.disable()
