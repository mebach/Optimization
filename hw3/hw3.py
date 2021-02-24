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
            g = (stressmax - stress)/stressmax
            g = np.append(g, (stress + stressmax)/stressmax)
            g[8] = (75e3 - stress[8]) / 75e3
            g[18] = (75e3 + stress[8]) / 75e3
            # print('g = ', g)
            return f, g
        f, g = func(x)
        # df, dg = finite_difference(func, x, 1e-6)
        df, dg = complex_step(func, x, 1e-12)
        # print('At this point, x = ', x)
        # print('At this next point, dg = ', dg)
        return f, g, df, dg

    xlast = np.array([])
    flast = np.array([])
    glast = np.array([])
    dflast = np.array([])
    dglast = np.array([])

    def obj(x):
        nonlocal xlast, flast, glast, dflast, dglast
        # print('Calling obj... \n')
        if not np.array_equal(x, xlast):
            flast, glast, dflast, dglast = objcon(x)
            xlast = x
        return flast, dflast

    def con(x):
        nonlocal xlast, flast, glast, dflast, dglast
        # print('Calling con... \n')
        if not np.array_equal(x, xlast):
            flast, glast, dflast, dglast = objcon(x)
            xlast = x
        return glast

    def jac(x):
        nonlocal xlast, flast, glast, dflast, dglast
        # print('Calling jac... \n')
        if not np.array_equal(x, xlast):
            flast, glast, dflast, dglast = objcon(x)
            xlast = x
        print('dglast = ', dglast)
        return dglast

    # ----------------------------------------------------

    def finite_difference(fun, x, h):
        # print('Calling finite_difference... \n')
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

    x0 = np.ones(10)
    # x0 = np.array([7.89999955, 0.1, 8.09999937, 3.9, 0.1, 0.1, 5.79827561, 5.51543289, 3.67695526, 0.14142136])
    # constraints = {'type': 'ineq', 'fun': con}
    lg = -np.inf
    ug = 0.0
    constraints = NonlinearConstraint(con, lg, ug, jac=jac)
    options = {'disp': True}

    # print('About to optimize... \n')
    res = minimize(obj, x0, bounds=Bounds(0.1, 10, keep_feasible=True), constraints=constraints, options=options, jac=True, method='slsqp')
    # print("x = ", res.x)
    # print('f = ', res.fun)
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
