import numpy as np
from scipy.optimize import minimize
from control.matlab import c2d, StateSpace

m = 5
k = 3
b = 0.5
H = 100  # number of points in the horizon

# define the state space equations of the form xdot = Ax + Bu
A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])

# convert the continuous state space system to a discrete time system
statespace = StateSpace(A, B, C, D)

# The interval of time between each discrete interval
Ts = 0.05

# creates an object which contains the discretized version of A, B, C, and D as well as a dt
discrete = c2d(statespace, Ts, method='zoh', prewarp_frequency=None)
objhist = np.ones(H)

def objcon(x):
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


def complex_step(fun, x, h):
    f, g = fun(x)
    # print(f, g)
    Jf = np.array([])
    Jg = np.zeros((len(g), len(x)))
    for i in range(len(x)):
        x_next = np.copy(x) + complex(0, 0)
        x_next[i] = x_next[i] + complex(0, h)
        f_next, g_next = fun(x_next)
        Jf = np.append(Jf, f_next.imag / h)
        Jg[:, i] = g_next.imag / h
        # print(Jg[:, i])

    return Jf, Jg

def Lagrange(obj, con, x, lam):
    f = obj(x)
    h = con(x)
    L = f + np.transpose(h) * lam
    return L



def constrained_optimizer(obj, con, x):

    lam = np.ones(len(con))




    gradR = np.vstack(L_hess, gradH)
    s = np.linalg.solve(gradR, R)
    u = u + s