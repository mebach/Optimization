import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

# design variables
start = np.array([0.0, 1.0])
end = np.array([1.0, 0.0])
n = 48  # number of nodes
x = np.linspace(start[0], end[0], n)

# constants
uk = 0.3
h = start[1]


def obj(y):
    f = 0
    for i in range(n-1):
        dx = x[i+1]-x[i]
        dy = y[i+1]-y[i]
        f = f + np.sqrt(dx**2 + dy**2)/(np.sqrt(h - y[i+1] - uk * x[i+1]) + np.sqrt(h - y[i] - uk * x[i]))

    return f


def con(input):
    g = np.zeros(2)
    g[0] = 1 - input[0]
    g[1] = input[-1]
    return g


# initial guess for y points
y = np.zeros(n)
y[0] = 1.0

constraints = {'type': 'eq', 'fun': con}

y = minimize(obj, y, bounds=Bounds(0.0, 1.0), constraints=constraints).x

a = np.linspace(0,1,100)
b = -np.sqrt(1-(a-1)**2) + 1

plt.plot(x, y, 'r--', a, b)
plt.show()
