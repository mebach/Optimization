import numpy as np
from control.matlab import c2d, StateSpace


m = 5
k = 3
b = 0.5

A = np.array([[0, 1], [-k/m, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])
statespace = StateSpace(A, B, C, D)

Ts = 0.2

discrete = c2d(statespace, Ts, method='zoh', prewarp_frequency=None)

print(discrete)

