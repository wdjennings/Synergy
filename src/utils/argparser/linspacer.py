from typing import Union

import numpy as np


def linspacer(min: Union[float, int] = 0, max: Union[float, int] = 1, count: int = 100):
    """
    ArgParser arguments for linspace values.
    Allows converting the argument to array of linspaced values.

    :param min: (float or int) default minimum value
    :param max:  (float or int) default minimum value
    :param count: (int) default count value
    :return: (dict) to be passed to parser using parser.add_argument(**linspacer())
    """
    def linspace_converter(value_str: str) -> np.ndarray:
        value_iter = iter(value_str.split(' '))
        return np.linspace(next(value_iter, min), next(value_iter, max), next(value_iter, count))
    return {
        'type': linspace_converter,
        'default': np.linspace(min, max, count)
    }
