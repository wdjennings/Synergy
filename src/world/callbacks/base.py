class Callback:
    """
    Callback class triggered at each time step.
    """

    def on_event_occurred(self, network: 'Network'):
        """
        An event has occurred in the network.

        :param network: (Network) network to get data from
        :return: None
        """
        pass

    def should_stop_simulation(self, network: 'Network') -> bool:
        """
        Should the simulation stop running now.

        :param network: (Network) network to get data from
        :return: (bool) should the simulation stop
        """
        return False

    def on_simulation_started(self, network: 'Network'):
        """
        The simulation has started in the network.

        :param network: (Network) network to get data from
        :return: None
        """
        pass

    def on_simulation_finished(self, network: 'Network'):
        """
        The simulation has finished in the network.

        :param network: (Network) network to get data from
        :return: None
        """
        pass
