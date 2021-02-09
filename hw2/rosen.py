import numpy as np


def rosen(x):
    f = (1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    dfdx = np.array([2 * (200.0 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)])
    return f, dfdx
