import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from math import sin, cos, sqrt, pi
import truss


def runoptimization(params, stressmax):

    def objcon(x):
        mass, stress = truss.truss(x)
        f = mass
        g[0] = 1 - stress/stressmax
        g[1] = 1 + stress/stressmax
        return f, g

    xlast = []
    flast = []
    glast = []

    def obj(x):
        nonlocal xlast, flast, glast
        if not np.array_equal(x, xlast):
            flast, glast = objcon(x)
            xlast = x
        return flast

    def con(x):
        nonlocal xlast, flast, glast
        if not np.array_equal(x, xlast):
            flast, glast = objcon(x)
            xlast = x
        return glast

    x0 = 0.5 * np.ones(10)
    constraints = {'type': 'ineq', 'fun': con}
    options = {'disp': True}

    res = minimize(obj, x0, bounds=Bounds(0.1, 1000), constraints=constraints, options=options)
    print("x = ", res.x)
    print('f = ', res.fun)
    print(res.success)


if __name__ == '__main__':
    params = 0
    stressmax = 25e3

    runoptimization(params, stressmax)