import numpy as np
import matplotlib.pyplot as plt
import quad
import rosen


def optimizer(fun, x, tau):

    def find_step(x, p):
        a = 0.5
        while fun(x + a*p) > fun(x):
            a = a / 2.0
        return a

    i = 0  # to keep track of how many iterations this goes through

    while np.linalg.norm(fun_grad(x)) > tau:
        p = -fun_grad(x)  # find search direction (negative of the gradient)
        alpha = find_step(x, p)  # determine step length
        x = x + alpha*p  # x(k+1) = x(k) + alpha(k)*p(k) update design variables
        i += 1

    return x, i


if __name__ == '__main__':

    x = np.array([5.0, 5.0])
    tau = 1e-4
    beta = 3./2

    x, i = optimizer()
    print('The optimized x = ', x)
    print('The number of iterations was: ', i)











