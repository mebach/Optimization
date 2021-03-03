import numpy as np
from scipy.optimize import minimize, Bounds
from control.matlab import c2d, StateSpace

# minimize sum from n = 1 to H (x_ref - x)


def runoptimization():

    # define parameters
    m = 5
    k = 3
    b = 0.5
    H = 10  # number of points in the horizon

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

    def objcon(u, discrete):
        print('forces are:', u)
        f = 0  # the objective f to minimize is the difference between the reference x and the actual x, summed across each point in the time horizon
        z = 0.0  # initial position of the mass
        zdot = 0.0  # initial velocity of the mass
        zref = 1.0  # the commanded position for the mass
        x = np.array([[z], [zdot]])  # vectorize the state variables
        for i in range(len(u)):
            f = f + (zref - z)
            # print(discrete.A)
            # print(x)
            # print(discrete.B)
            # print(u[i])
            x_next = np.matmul(discrete.A, x) + np.matmul(discrete.B, np.array([u[i]]))
            z = x_next[0]
        return f

    ulast = []
    flast = []
    glast = []

    def obj(u, discrete):
        nonlocal ulast, flast, glast
        if not np.array_equal(u, ulast):
            flast = objcon(u, discrete)
            ulast = u
        return flast

    def con(u):
        nonlocal ulast, flast, glast
        if not np.array_equal(u, ulast):
            flast = objcon(u, discrete)
            ulast = u
        return glast

    u0 = 0.5 * np.ones(H)
    constraints = {'type': 'ineq', 'fun': con}
    options = {'disp': True}

    res = minimize(obj, u0, args=discrete, bounds=Bounds(-1.0, 1.0), options=options)
    # print("x = ", res.x)
    # print('f = ', res.fun)
    # print(res.success)
    # print(res.message)
    x_star = res.x
    return x_star

if __name__ == '__main__':
    x_star = runoptimization()
    print('the optimized forces are: ', x_star)