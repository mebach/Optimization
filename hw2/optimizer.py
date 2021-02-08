import numpy as np
import matplotlib.pyplot as plt


def optimizer(x, tau, beta):

    def rosen(x):
        return (1-x[0])**2 + 100.0*(x[1]-x[1]**2)**2

    def rosen_grad(x):
        return np.array([2*(200.0*x[0]**3 - 200*x[0]*x[1] + x[0] - 1), 200*(x[1]-x[0]**2)])

    def quad(x, beta):
        return x[0]**2 + x[1]**2 - beta*x[0]*x[1]

    def quad_grad(x, beta):
        return np.array([2*x[0] - beta*x[1], 2*x[1] - beta*x[0]])

    def find_step(x, p):
        a = 1.0
        while np.linalg.norm(quad(x + a*p, beta)) > np.linalg.norm(quad(x, beta)):
            a = a / 2.0
            print('Finding step...')
        return a

    i = 0  # to keep track of how many iterations this goes through
    while np.linalg.norm(quad_grad(x, beta)) > tau:
        print('i = ', i)
        p = -quad_grad(x, beta)  # find search direction (negative of the gradient)
        print('p = ', p)
        alpha = find_step(x, p)  # determine step length
        print('alpha =', alpha)
        x = x + alpha*p  # x(k+1) = x(k) + alpha(k)*p(k) update design variables
        i += 1

    return x, i


if __name__ == '__main__':
    x = np.array([2.0, -1.0])
    tau = 1e-4
    beta = 3./2

    x, i = optimizer(x, tau, beta)
    print('The optimized x = :', x)
    print('The number of iterations was: ', i)







