import numpy as np
from control.matlab import c2d, StateSpace
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, Bounds

def runoptimization():
    # define parameters
    mc = 1.0
    mr = 0.25
    g = 9.81
    Fe = (mc + 2 * mr) * g
    mu = 0.1
    Jc = 0.0042
    d = 0.3

    H = 7  # number of points in the horizon

    # define the state space equations of the form xdot = Ax + Bu
    A_long = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_long = np.array([[0], [1 / (mc + 2 * mr)]])
    C_long = np.array([[1, 0], [0, 1]])
    D_long = np.array([[0], [0]])

    A_lat = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, -(Fe) / (mc + 2 * mr), (-mu) / (mc + 2 * mr), 0],
                      [0, 0, 0, 0]])
    B_lat = np.array([[0],
                      [0],
                      [0],
                      [1 / (Jc + 2 * mr * d ** 2)]])
    C_lat = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
    D_lat = np.array([[0],
                      [0]])

    # convert the continuous state space system to a discrete time system
    statespace_long = StateSpace(A_long, B_long, C_long, D_long)
    statespace_lat = StateSpace(A_lat, B_lat, C_lat, D_lat)

    # The interval of time between each discrete interval
    Ts = 0.2

    # creates an object which contains the discretized version of A, B, C, and D as well as a dt
    discrete_long = c2d(statespace_long, Ts, method='zoh', prewarp_frequency=None)
    discrete_lat = c2d(statespace_lat, Ts, method='zoh', prewarp_frequency=None)
    objhist_h = np.ones(H)
    objhist_z = np.ones(H)

    def objcon_long(u, discrete_long):
        f = 0  # the objective f to minimize is the difference between the reference x and the actual x, summed across each point in the time horizon
        h = 0.0  # initial position of the mass
        hdot = 0.0  # initial velocity of the mass

        href = 1.0  # the commanded longitudinal position for the VTOL

        x_long = np.array([[h], [hdot]])  # vectorize the state variables

        # g = np.array([0])
        for i in range(len(u)):
            objhist_h[i] = h
            # if i != 0:
            #     g = np.append(g, 100 - (u[i] - u[i-1])/Ts)
            #     g = np.append(g, 100 + (u[i] - u[i-1])/Ts)
            f = f + abs(href - h)

            x_long_next = np.matmul(discrete_long.A, x_long) + np.matmul(discrete_long.B, np.array([[u[i]]]))

            h = x_long_next[0]
            x_long = x_long_next
        return f[0, 0]

    def objcon_lat(u, discrete_lat):
        f = 0  # the objective f to minimize is the difference between the reference x and the actual x, summed across each point in the time horizon
        z = 0.0
        theta = 0.0
        zdot = 0.0
        thetadot = 0.0

        zref = 2.0  # the commanded lateral position for the VTOL
        x_lat = np.array([[z], [theta], [zdot], [thetadot]])
        # g = np.array([0])
        for i in range(len(u)):
            objhist_z[i] = z
            # if i != 0:
            #     g = np.append(g, 100 - (u[i] - u[i-1])/Ts)
            #     g = np.append(g, 100 + (u[i] - u[i-1])/Ts)
            f = f + abs(zref - z)

            x_lat_next = np.matmul(discrete_lat.A, x_lat) + np.matmul(discrete_lat.B, np.array([[u[i]]]))
            z = x_lat_next[0]
            x_lat = x_lat_next
        return f[0, 0]

    ulast_long = []
    flast_long = []
    glast_long = []
    ulast_lat = []
    flast_lat = []
    glast_lat = []

    def obj_long(u, discrete_long):
        nonlocal ulast_long, flast_long, glast_long
        if not np.array_equal(u, ulast_long):
            flast_long = objcon_long(u, discrete_long)
            ulast_long = u
        return flast_long

    def con_long(u):
        nonlocal ulast_long, flast_long, glast_long
        if not np.array_equal(u, ulast_long):
            flast_long = objcon_long(u, discrete_long)
            ulast_long = u
        return glast_long

    def obj_lat(u, discrete_lat):
        nonlocal ulast_lat, flast_lat, glast_lat
        if not np.array_equal(u, ulast_lat):
            flast_lat = objcon_lat(u, discrete_lat)
            ulast_lat = u
        return flast_lat

    def con_lat(u, discrete_lat):
        nonlocal ulast_lat, flast_lat, glast_lat
        if not np.array_equal(u, ulast_lat):
            flast_lat = objcon_lat(u, discrete_lat)
            ulast_lat = u
        return glast_lat

    def nelder_mead(fun, discrete, x, tau_x, tau_f):
        # initialize variables
        n = len(x)
        simplex = np.zeros((n, n+1))
        simplex[:, 0] = x
        s = np.zeros(n)
        l = 0.0001

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
        while gradx(simplex, n) > tau_x or gradf(fun, simplex, n) > tau_f:
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

    u0_long = 0.5 * np.ones(H)
    u0_lat = 0.5 * np.ones(H)
    constraints = {'type': 'ineq', 'fun': con_long}
    options = {'disp': False}

    x = nelder_mead(obj_long, discrete_long, u0_long, 1e-6, 1e-8)
    print('Optimized x: ', x)
    plt.plot(objhist_h)
    plt.xlabel('Time (s)')
    plt.ylabel('Position z (m)')

    lb = -100.0
    ub = 100.0
    bounds = Bounds([lb, lb, lb, lb, lb, lb, lb], [ub, ub, ub, ub, ub, ub, ub])
    res_long = differential_evolution(obj_long, bounds=bounds, args=(discrete_long,), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=0.5, recombination=0.7, disp=True, polish=True, init='latinhypercube', atol=0)
    res_lat = differential_evolution(obj_lat, bounds=bounds, args=(discrete_lat,), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=0.5, recombination=0.7, disp=True, polish=True, init='latinhypercube', atol=0)
    print('Optimized x_gen: ', res_long.x)
    print('Optimized tau_gen: ', res_lat.x)
    plt.plot(objhist_h)
    # plt.plot(objhist_z)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.show()



if __name__ == '__main__':
    runoptimization()


