import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import time

# design variables
start = np.array([0.0, 1.0])
end = np.array([1.0, 0.0])
n = 12  # number of nodes
x = np.linspace(start[0], end[0], n)

# constants
uk = 0.3
h = start[1]

# counters
obj_calls = 0
minimize_time = 0


def obj(y):
    f = 0
    y = np.insert(y, 0, start[1])
    y = np.append(y, end[1])
    for i in range(n-1):
        dx = x[i+1]-x[i]
        dy = y[i+1]-y[i]
        f = f + (np.sqrt(dx**2 + dy**2)/(np.sqrt(h - y[i+1] - uk*x[i+1]) + np.sqrt(h - y[i] - uk*x[i])))
        if math.isnan(f):
            print('y[i+1] = ', y[i+1])
            print('x[i+1] = ', x[i+1])
            print('f = ', f)
    return f


# initial guess for y points
y_guess = np.linspace(1, 0, n)
y_guess = y_guess[1:n-1]

options = {'eps': 1e-6}

minimize_time = time.time()
y_star = np.array(minimize(obj, y_guess, options=options).x)
minimize_time = time.time() - minimize_time
y_star = np.insert(y_star, 0, start[1])
y_star = np.append(y_star, end[1])

dt = 0  # travelling time counter
for i in range(n-1):
    dx = x[i + 1] - x[i]
    dy = y_star[i + 1] - y_star[i]
    dt = dt + (np.sqrt(2/9.81))*(np.sqrt(dx ** 2 + dy ** 2) / (np.sqrt(h - y_star[i + 1] - uk * x[i + 1]) + np.sqrt(h - y_star[i] - uk * x[i])))

print('The total travelling time is: ', dt)
print('The total time to optimize is: ', minimize_time)
print('The number of function calls is: ', obj_calls)

a = np.linspace(0, 1, 100)
b = -np.sqrt(1-(a-1)**2) + 1

plt.plot(x, y_star)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
# plt.axis('equal')
plt.show()
