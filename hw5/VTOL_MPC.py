import numpy as np
from scipy.optimize import minimize, Bounds
from control.matlab import c2d, StateSpace
import matplotlib.pyplot as plt
import time

def runoptimization():

    # define parameters
    mc = 1.0
    mr = 0.25
    g = 9.81
    Fe = (mc + 2*mr)*g
    mu = 0.1
    Jc = 0.0042
    d = 0.3

    H = 50  # number of points in the horizon

    # define the state space equations of the form xdot = Ax + Bu
    A_long = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_long = np.array([[0], [1 / (mc + 2*mr)]])
    C_long = np.array([[1, 0], [0, 1]])
    D_long = np.array([[0], [0]])

    A_lat = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, -(Fe)/(mc+2*mr), (-mu)/(mc+2*mr), 0],
                      [0, 0, 0, 0]])
    B_lat = np.array([[0],
                      [0],
                      [0],
                      [1/(Jc+2*mr*d**2)]])
    C_lat = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
    D_lat = np.array([[0],
                      [0]])


    # convert the continuous state space system to a discrete time system
    statespace_long = StateSpace(A_long, B_long, C_long, D_long)
    statespace_lat = StateSpace(A_lat, B_lat, C_lat, D_lat)

    # The interval of time between each discrete interval
    Ts = 0.05

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

        zref = 1.0  # the commanded lateral position for the VTOL
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


    u0_long = 0.5 * np.ones(H)
    u0_lat = 0.5 * np.ones(H)
    constraints = {'type': 'ineq', 'fun': con_long}
    options = {'disp': False}

    t0 = time.time()
    res_f = minimize(obj_long, u0_long, args=discrete_long, bounds=Bounds(-10.0, 10.0), options=options, method='slsqp')
    res_tau = minimize(obj_lat, u0_lat, args=discrete_lat, bounds=Bounds(-10.0, 10.0), options=options, method='slsqp')
    t1 = time.time() - t0
    print('Time elapsed:', t1)
    # print("x = ", res.x)
    # print('f = ', res.fun)
    # print(res.success)
    # print(res.message)
    x_long_star = res_f.x
    x_lat_star = res_tau.x
    return x_long_star, x_lat_star, objhist_h, objhist_z


if __name__ == '__main__':
    x_long_star, x_lat_star, objhist_h, objhist_z = runoptimization()
    print('the optimized forces are: ', x_long_star)
    print('the optimized torques are: ', x_lat_star)
    t = 0.05 * np.array(range(50))
    plt.plot(t, objhist_h)
    plt.plot(t, objhist_z)
    plt.xlabel('Time (s)')
    plt.ylabel('Position h (m)')
    plt.show()