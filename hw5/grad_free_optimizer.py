import numpy as np
from control.matlab import c2d, StateSpace
import matplotlib.pyplot as plt

def runoptimization():
    # define parameters
    m = 5
    k = 3
    b = 0.5
    H = 3  # number of points in the horizon

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

    def objcon(u, discrete):
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

    ulast = []
    flast = []
    glast = []

    def obj(u, discrete):
        nonlocal ulast, flast, glast
        if not np.array_equal(u, ulast):
            flast, glast = objcon(u, discrete)
            ulast = u
        return flast

    def con(u):
        nonlocal ulast, flast, glast
        if not np.array_equal(u, ulast):
            flast, glast = objcon(u, discrete)
            ulast = u
        return glast

    def nelder_mead(fun, discrete, x, tau_x, tau_f):
        # initialize variables
        n = len(x)
        simplex = np.zeros((n, n+1))
        simplex[:, 0] = x
        s = np.zeros(n)
        l = 1

        def gradx(simplex0, n):
            sum = 0
            for i in range(n):
                sum += np.linalg.norm(simplex0[:, i] - simplex0[:, n])
            return sum

        def gradf(fun, simplex0, n):
            sum = 0
            for i in range(n):
                sum += (fun(simplex0[:, i], discrete) - fun(simplex0[:, n], discrete))**2
            return np.sqrt(sum/(n+1))

        for j in range(n):
            for i in range(n):
                if j == i:
                    s[i] = (l/(n*np.sqrt(2)))*(np.sqrt(n+1)-1) + (1/np.sqrt(2))
                else:
                    s[i] = (l/(n*np.sqrt(2)))*(np.sqrt(n+1)-1)
            simplex[:, j+1] = x + s

        f = np.zeros(n+1)
        while gradx(simplex, n) > tau_x or gradf(fun, simplex, n):
            for i in range(n+1):
                f[i] = fun(simplex[:, i], discrete)
            simplex = simplex[:, np.argsort(f)]
            xc = (1/n) * (simplex.sum(axis=1)-simplex[:, -1])
            xr = xc + (xc - simplex[:, -1])
            if fun(xr, discrete) < fun(simplex[:, 0], discrete):
                xe = xc + 2*(xc - simplex[:, -1])
                if fun(xe, discrete) < fun(simplex[:, 0], discrete):
                    simplex[:, -1] = xe
                else:
                    simplex[:, -1] = xr
            elif fun(xr, discrete) <= fun(simplex[:, -2], discrete):
                simplex[:, -1] = xr
            else:
                if fun(xr, discrete) > fun(simplex[:, -1], discrete):
                    xic = xc - 0.5 * (xc - simplex[:, -1])
                    if fun(xic, discrete) < fun(simplex[:, -1], discrete):
                        simplex[:, -1] = xic
                    else:
                        for i in range(n):
                            simplex[:, i+1] = simplex[:, 0] + 0.5 * (simplex[:, i+1]-simplex[:, 0])
                else:
                    xoc = xc + 0.5 * (xc - simplex[:, -1])
                    if fun(xoc, discrete) < fun(xr, discrete):
                        simplex[:, -1] = xoc
                    else:
                        for i in range(n):
                            simplex[:, i+1] = simplex[:, 0] + 0.5 * (simplex[:, i+1] - simplex[:, 0])

        return simplex[:,0]

    u0 = 0.5 * np.ones(H)

    x = nelder_mead(obj, discrete, u0, 1.0, 1.0)
    print('Optimized x: ', x)
    plt.plot(objhist)
    plt.xlabel('Time (s)')
    plt.ylabel('Position z (m)')
    plt.show()


if __name__ == '__main__':
    runoptimization()


