import numpy as np
from truss import truss


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
    # A = np.ones(10)
    A = np.array([1.02608886, 0.45321982, 1.12683251, 1.53518411, 0.1, 0.45321983, 1.18241006, 0.9771986, 1.57490896, 0.45498688])
    Jf, Jg = complex_step(truss, A, 1e-6)
    print('Jf = ', Jf)
    print('Jg = ', Jg)
