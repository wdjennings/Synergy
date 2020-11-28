# from src.utils.logging import use_logging_debug_mode
# from src.world.callbacks.base import Callback
#
#
# class LoggingCallback(Callback):
#
#     def __init__(self):
#         pass
#
#     def on_simulation_started(self, network: 'Network'):
#         use_logging_debug_mode()
#
#     def on_event_occurred(self, network: 'Network'):
#         # TODO event arg
#
#         if event.type == EventType.extinction_event:
#             logger.info('[time = %.2f] No more infected cells' % network.time)
#
#         elif event.type == EventType.remove_event:
#             logger.info('[time = %.2f] Removed cell {}'.format(cell_to_remove) % network.time)
#
#         elif event.type == EventType.infection_event:
#             logger.info('[time = %.2f] Infected cell {}'.format(cell_to_infect) % network.time)