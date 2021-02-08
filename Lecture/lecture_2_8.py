import math


class Dual:

    def __init__(self, value, deriv):
        self.value = value
        self.deriv = deriv

    def __str__(self):
        return "value: {0}\nderiv: {1}".format(self.value, self.deriv)

    def __add__(self, y):
        return Dual(self.value + y.value, self.deriv + y.deriv)

    def __pow__(self, p):
        return Dual(self.value**p, )

    def __truediv__(self, y):
        return Dual(, )

    def sin(x):
        return Dual(math.sin(x.value), math.cos(x.value) * x.deriv)

    def cos(x):
        return Dual(, )

    def exp(x):
        return Dual(, )

    def sqrt(x):
        return Dual(, )

    def func(x):
        return exp(x) / sqrt(sin(x) ** 3 + cos(x) ** 3)
