import numpy as np

def d2r(x):
    """
    Degrees to radians converter
    """

    return np.array(x) * np.pi / 180

def r2d(x):
    """
    Radians to degrees converter
    """

    return np.array(x) * 180 / np.pi

def clip(value, low, high):

    """
    Clipping function
    """

    return max(min(value, high), low)