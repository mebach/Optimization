import numpy as np
from scipy.optimize import minimize, Bounds
from control.matlab import c2d, StateSpace
import matplotlib.pyplot as plt


def runOptimization():
    m = 5
    k = 3
    b = 0.5
    H = 20  # number of points in the horizon

    # define the state space equations of the form xdot = Ax + Bu
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1 / m]])
    C = np.array([[1, 0], [0, 1]])
    D = np.array([[0], [0]])

    # convert the continuous state space system to a discrete time system
    statespace = StateSpace(A, B, C, D)

    # The interval of time between each discrete interval
    Ts = 0.2

    # creates an object which contains the discretized version of A, B, C, and D as well as a dt
    discrete = c2d(statespace, Ts, method='zoh', prewarp_frequency=None)
    objhist = np.ones(H)

    def objcon(u):
        f = 0  # the objective f to minimize is the difference between the reference x and the actual x, summed across each point in the time horizon
        z = 0.0  # initial position of the mass
        zdot = 0.0  # initial velocity of the mass
        zref = 1.0  # the commanded position for the mass
        x = np.array([[z], [zdot]])  # vectorize the state variables
        g = np.array([0])
        for i in range(len(u)):
            objhist[i] = z
            if i != 0:
                g = np.append(g, 100 - (u[i] - u[i - 1]) / Ts)
                g = np.append(g, 100 + (u[i] - u[i - 1]) / Ts)
            f = f + abs(zref - z)
            x_next = np.matmul(discrete.A, x) + np.matmul(discrete.B, np.array([[u[i]]]))
            z = x_next[0]
            x = x_next
        return f[0, 0], g


    def penalty(x, mu):
        f, g = objcon(x)

        sum = 0
        for i in range(len(g)):
            sum = sum + min(0, g[i])**2
        return f + (mu/2 * sum)


    xlast = np.array([])
    flast = np.array([])
    glast = np.array([])
    dflast = np.array([])
    dglast = np.array([])

    def obj(x):
        nonlocal xlast, flast, glast, dflast, dglast
        if not np.array_equal(x, xlast):
            flast, glast = objcon(x)
            xlast = x
        return flast

    def con(x):
        nonlocal xlast, flast, glast, dflast, dglast
        if not np.array_equal(x, xlast):
            flast, glast = objcon(x)
            xlast = x
        return glast


    def constrained_optimizer(fun, x):
        mu = 1.0
        rho = 1.1
        k = 0
        convergence = 1e-2
        f_next = 100
        f = 0

        options = {'disp': True}
        while f_next-f > convergence:
            f = f_next
            res = minimize(fun, x, args=mu, bounds=Bounds(-10.0, 10.0), options=options, method='slsqp')
            mu = rho * mu
            x = res.x
            f_next = res.fun
            print('The forces are: ', x)
            print('mu = ', mu)
            k += 1
            print('k = ', k)
            print('f = ', f)
            print('f_next = ', f_next)
        return x


    x0 = np.ones(H)
    x = constrained_optimizer(penalty, x0)

    f = obj(x)
    print('x = ', x)
    print('f = ', f)
    t = Ts * np.array(range(H))
    plt.plot(t, objhist)
    plt.show()


if __name__ == '__main__':
    runOptimization()
