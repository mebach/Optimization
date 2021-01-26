import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from math import sin, cos, sqrt, pi
import truss
import numpy as np


def runoptimization(params, stressmax):

    def objcon(x):
        mass, stress = truss.truss(x)
        f = mass/100
        g = 1.0 - stress/stressmax
        g = np.append(g, stress/stressmax + 1)
        g[8] = 1.0 - stress[8]/75e3
        g[18] = 1.0 + stress[8] / 75e3
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

    res = minimize(obj, x0, bounds=Bounds(0.1, 10), constraints=constraints, options=options)
    # print("x = ", res.x)
    # print('f = ', res.fun)
    # print(res.success)
    # print(res.message)
    x_star = res.x
    return x_star


if __name__ == '__main__':
    params = 0
    stressmax = 25e3

    x_star = runoptimization(params, stressmax)

    mass, stress = truss.truss(x_star)

    print('The optimized diameters are: ', x_star)
    print('Mass is: ', mass)
    print('Stress in each member is: ', stress)

