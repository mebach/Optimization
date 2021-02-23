import numpy as np
from truss_AD import truss_AD
import algopy


def AD(fun, x):
    x_dual = algopy.UTPM.init_jacobian(x)  # essentially, this function just converts the input cross sectional areas into algopy's version of dual numbers
    mass, stress = fun(x_dual)
    Jf = algopy.UTPM.extract_jacobian(mass)
    Jg = algopy.UTPM.extract_jacobian(stress)

    return Jf, Jg


if __name__ == '__main__':
    A = np.ones(10)
    Jf, Jg = AD(truss_AD, A)
    print('Jf = ', Jf)
    print('Jg = ', Jg)

