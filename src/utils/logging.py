
def use_logging_debug_mode():
    """ Set logging module to use debug mode (also prints to stdout) """
    import logging as logging_module
    import sys
    logging_module.basicConfig(stream=sys.stdout, level=logging_module.DEBUG)
