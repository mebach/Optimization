import numpy as np


def quad(x):
    f = x[0] ** 2 + x[1] ** 2 - 3.0 / 2 * x[0] * x[1]
    dfdx = np.array([2 * x[0] - 3.0 / 2 * x[1], 2 * x[1] - 3.0 / 2 * x[0]])
    return f, dfdx
