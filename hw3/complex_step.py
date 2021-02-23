import numpy as np
from truss import truss
import cmath


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


if __name__ == '__main__':
    A = 2* np.ones(10)
    Jf, Jg = complex_step(truss, A, 1e-6)
    print('Jf = ', Jf)
    print('Jg = ', Jg)
