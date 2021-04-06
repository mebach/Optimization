import numpy as np
from scipy.optimize import minimize, Bounds
from control.matlab import c2d, StateSpace
import matplotlib.pyplot as plt
import time

class VTOL_MPC():

    def __init__(self):
    # define parameters
        self.mc = 1.0
        self.mr = 0.25
        self.g = 9.81
        self.Fe = (self.mc + 2*self.mr)*self.g
        self.mu = 0.1
        self.Jc = 0.0042
        self.d = 0.3

        self.H = 8  # number of points in the horizon

        # define the state space equations of the form xdot = Ax + Bu
        self.A_long = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.B_long = np.array([[0], [1 / (self.mc + 2*self.mr)]])
        self.C_long = np.array([[1, 0], [0, 1]])
        self.D_long = np.array([[0], [0]])

        self.A_lat = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0, -(self.Fe)/(self.mc+2*self.mr), (-self.mu)/(self.mc+2*self.mr), 0],
                          [0, 0, 0, 0]])
        self.B_lat = np.array([[0],
                          [0],
                          [0],
                          [1/(self.Jc+2*self.mr*self.d**2)]])
        self.C_lat = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
        self.D_lat = np.array([[0],
                          [0]])


        # convert the continuous state space system to a discrete time system
        self.statespace_long = StateSpace(self.A_long, self.B_long, self.C_long, self.D_long)
        self.statespace_lat = StateSpace(self.A_lat, self.B_lat, self.C_lat, self.D_lat)

        # The interval of time between each discrete interval
        self.Ts = 0.5

        # creates an object which contains the discretized version of A, B, C, and D as well as a dt
        self.discrete_long = c2d(self.statespace_long, self.Ts, method='zoh', prewarp_frequency=None)
        self.discrete_lat = c2d(self.statespace_lat, self.Ts, method='zoh', prewarp_frequency=None)
        self.objhist_h = np.ones(self.H)
        self.objhist_z = np.ones(self.H)

        self.ulast_long = []
        self.flast_long = []
        self.glast_long = []
        self.ulast_lat = []
        self.flast_lat = []
        self.glast_lat = []

    def objcon_long(self, u, r, x):
        f = 0  # the objective f to minimize is the difference between the reference x and the actual x, summed across each point in the time horizon
        h = x.item(1)  # current longitudinal position of the VTOL
        hdot = x.item(4)  # current longitudinal velocity of the VTOL

        href = r.item(1)  # the commanded longitudinal position for the VTOL

        x_long = np.array([[h], [hdot]])  # vectorize the state variables

        # g = np.array([0])
        for i in range(len(u)):
            self.objhist_h[i] = h
            # if i != 0:
            #     g = np.append(g, 100 - (u[i] - u[i-1])/Ts)
            #     g = np.append(g, 100 + (u[i] - u[i-1])/Ts)
            f = f + abs(href - h)

            x_long_next = np.matmul(self.discrete_long.A, x_long) + np.matmul(self.discrete_long.B, np.array([[u[i]]]))

            h = x_long_next[0]
            x_long = x_long_next
        return f[0, 0]

    def objcon_lat(self, u, r, x):
        f = 0  # the objective f to minimize is the difference between the reference x and the actual x, summed across each point in the time horizon
        z = x.item(0)
        theta = x.item(2)
        zdot = x.item(3)
        thetadot = x.item(5)

        zref = r.item(0)  # the commanded lateral position for the VTOL
        x_lat = np.array([[z], [theta], [zdot], [thetadot]])
        # g = np.array([0])
        for i in range(len(u)):
            self.objhist_z[i] = z
            # if i != 0:
            #     g = np.append(g, 100 - (u[i] - u[i-1])/Ts)
            #     g = np.append(g, 100 + (u[i] - u[i-1])/Ts)
            f = f + abs(zref - z)

            x_lat_next = np.matmul(self.discrete_lat.A, x_lat) + np.matmul(self.discrete_lat.B, np.array([[u[i]]]))
            z = x_lat_next[0]
            x_lat = x_lat_next
        return f[0, 0]



    def obj_long(self, u, r, x):
        # nonlocal ulast_long, flast_long, glast_long
        if not np.array_equal(u, self.ulast_long):
            self.flast_long = self.objcon_long(u, r, x)
            self.ulast_long = u
        return self.flast_long

    def con_long(self, u, r, x):
        # nonlocal ulast_long, flast_long, glast_long
        if not np.array_equal(u, self.ulast_long):
            self.flast_long = self.objcon_long(u, r, x)
            self.ulast_long = u
        return self.glast_long

    def obj_lat(self, u, r, x):
        # nonlocal ulast_lat, flast_lat, glast_lat
        if not np.array_equal(u, self.ulast_lat):
            self.flast_lat = self.objcon_lat(u, r, x)
            self.ulast_lat = u
        return self.flast_lat

    def con_lat(self, u, r, x):
        # nonlocal ulast_lat, flast_lat, glast_lat
        if not np.array_equal(u, self.ulast_lat):
            self.flast_lat = self.objcon_lat(u, r, x)
            self.ulast_lat = u
        return self.glast_lat

    def update(self, r, x):
        u0_long = 0.5 * np.ones(self.H)
        u0_lat = 0.5 * np.ones(self.H)
        constraints = {'type': 'ineq', 'fun': self.con_long}
        options = {'disp': False}

        # t0 = time.time()
        res_f = minimize(self.obj_long, u0_long, args=(r, x), bounds=Bounds(-5.0, 5.0), options=options, method='slsqp')
        res_tau = minimize(self.obj_lat, u0_lat, args=(r, x), bounds=Bounds(-0.5, 0.5), options=options, method='slsqp')
        # t1 = time.time() - t0
        # print('Time elapsed:', t1)
        # print("x = ", res.x)
        # print('f = ', res.fun)
        # print(res.success)
        # print(res.message)
        x_long_star = res_f.x
        x_lat_star = res_tau.x
        return np.array([[res_f.x.item(0)+self.Fe],[res_tau.x.item(0)]])
