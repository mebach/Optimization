import numpy as np
from truss import truss


def getJacobian(fun, x, h):
    f, g = fun(x)
    print(f, g)
    Jf = np.array([])
    Jg = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        x_next = np.copy(x)
        x_next[i] = x_next[i] + h
        f_next, g_next = fun(x_next)
        Jf = np.append(Jf, (f_next - f)/h)
        Jg[:, i] = (g_next - g)/h
        print(Jg[:, i])

    return Jf, Jg


if __name__ == '__main__':
    A = 3 * np.ones(10)
    Jf, Jg = getJacobian(truss, A, 0.1)
    print('Jf = ', Jf)
    print('Jg = ', Jg)
