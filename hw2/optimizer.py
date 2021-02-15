import numpy as np
import matplotlib.pyplot as plt
from quad import quad as qd
from rosen import rosen as ros
from brach import brachistochrone as br
from scipy.optimize import minimize


def optimizer(fun, x, tau):

    def find_step(x, p, fun_calls):
        a = 0.1
        f1, g1 = fun(x)
        fun_calls += 1
        f2, g2 = fun(x + a*p)
        fun_calls += 1
        while f2 > f1:
            a = a * 0.5
            f2, g2 = fun(x + a * p)

        return a

    i = 0  # to keep track of how many iterations this goes through
    fun_calls = 0  # to keep track of how many function calls are required
    g = np.array([])
    p = np.zeros(len(x)) # initialize the search direction

    f, grad = fun(x)
    fun_calls += 1
    B = np.matmul(np.transpose(grad),grad)

    while np.linalg.norm(grad) > tau and i < 10000:
        g = np.append(g, np.linalg.norm(grad))
        p = -grad + B * p  # find search direction (negative of the gradient)
        alpha = find_step(x, p, fun_calls)  # determine step length
        x = x + alpha*p  # x(k+1) = x(k) + alpha(k)*p(k) update design variables

        f_new, grad_new = fun(x)  # recalculate the new function
        fun_calls += 1
        B = np.matmul(np.transpose(grad_new), grad_new) / np.matmul(np.transpose(grad), grad)  # recalculate the new damping parameter
        f = f_new
        grad = grad_new
        i += 1

    return x, i, fun_calls, g


if __name__ == '__main__':

    # x0 = np.array([5.0, 5.0])
    x0 = 0.5 * np.zeros(58)
    tau = 1e-5
    beta = 3./2

    x, i, fun_calls, g = optimizer(br, x0, tau)
    print('The optimized x = ', x)
    print('The number of iterations was: ', i)
    print('The number of function calls was: ', fun_calls)

    x = np.concatenate([[1.0], x, [0.0]])
    t = np.linspace(0.0, 1.0, len(x))
    # plt.plot(t, x)
    plt.plot(g)
    plt.xlabel('Optimizer Iterations')
    plt.ylabel('Norm of the Gradient')
    plt.yscale('log')
    plt.title('Brachistochrone Function')
    plt.show()










