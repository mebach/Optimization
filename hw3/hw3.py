from scipy.optimize import minimize
from scipy.optimize import Bounds, NonlinearConstraint
from truss import truss
from truss_AD import truss_AD
import algopy
import numpy as np


def runoptimization(stressmax):
    def objcon(x):
        def func(x):
            mass, stress = truss(x)
            f = mass / 100
            file.write(str(flast * 100))
            file.write('\n')
            g = (stressmax - stress)/stressmax
            g = np.append(g, (stress + stressmax)/stressmax)
            g[8] = (75e3 - stress[8]) / 75e3
            g[18] = (75e3 + stress[8]) / 75e3
            return f, g
        f, g = func(x)
        df, dg = complex_step(func, x, 1e-12)
        return f, g, df, dg

    xlast = np.array([])
    flast = np.array([])
    glast = np.array([])
    dflast = np.array([])
    dglast = np.array([])

    def obj(x):
        nonlocal xlast, flast, glast, dflast, dglast
        if not np.array_equal(x, xlast):
            flast, glast, dflast, dglast = objcon(x)
            xlast = x
        return flast, dflast

    def con(x):
        nonlocal xlast, flast, glast, dflast, dglast
        if not np.array_equal(x, xlast):
            flast, glast, dflast, dglast = objcon(x)
            xlast = x
        return glast

    def jac(x):
        nonlocal xlast, flast, glast, dflast, dglast
        if not np.array_equal(x, xlast):
            flast, glast, dflast, dglast = objcon(x)
            xlast = x
        return dglast

    # ----------------------------------------------------

    def finite_difference(fun, x, h):
        f, g = fun(x)
        Jf = np.array([])
        Jg = np.zeros((len(g), len(x)))
        for i in range(len(x)):
            x_next = np.copy(x)
            x_next[i] = x_next[i] + h
            f_next, g_next = fun(x_next)
            Jf = np.append(Jf, (f_next - f) / h)
            Jg[:, i] = (g_next - g) / h
        return Jf, Jg

    def complex_step(fun, x, h):
        f, g = fun(x)
        Jf = np.array([])
        Jg = np.zeros((len(g), len(x)))
        for i in range(len(x)):
            x_next = np.copy(x) + complex(0, 0)
            x_next[i] = x_next[i] + complex(0, h)
            f_next, g_next = fun(x_next)
            Jf = np.append(Jf, f_next.imag / h)
            Jg[:, i] = g_next.imag / h
        return Jf, Jg

    def AD(fun, x):
        x_dual = algopy.UTPM.init_jacobian(x)  # essentially, this function just converts the input cross sectional areas into algopy's version of dual numbers
        mass, stress = fun(x_dual)
        Jf = algopy.UTPM.extract_jacobian(mass)
        Jg = algopy.UTPM.extract_jacobian(stress)
        return Jf, Jg

    # ----------------------------------------------------

    x0 = 0.2 * np.ones(10)
    lg = 0.0
    ug = np.inf
    constraints = NonlinearConstraint(con, lg, ug, jac=jac)
    options = {'disp': True}

    res = minimize(obj, x0, bounds=Bounds(0.1, 10, keep_feasible=True), constraints=constraints, options=options, jac=True, method='slsqp')
    print(res.success)
    print(res.message)
    x_star = res.x
    return x_star


if __name__ == '__main__':
    file = open('mass_file.txt', 'w')

    params = 0
    stressmax = 25e3

    x_star = runoptimization(stressmax)

    mass, stress = truss(x_star)

    print('The optimized diameters are: ', x_star)
    print('Mass is: ', mass)
    print('Stress in each member is: ', stress)

