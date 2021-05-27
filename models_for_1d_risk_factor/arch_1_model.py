"""ARCH(1) model:

r_t = log(p_t) - log(p_(t-1))
r_t = s_t * e_t
s_t^2 = w + a * r_t^2
"""
from pprint import pprint
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from time_series_model_template import TimeSeriesModel


class Arch1(TimeSeriesModel):
    def __init__(self):
        self.omega = None
        self.alpha = None

    def fit_parameters(self, dt, x):
        # least square method
        # E[r_t^2] = w + a * r_(t-1)^2
        r2 = x ** 2
        ones = np.ones_like(r2[1:])
        opt_result = lsq_linear(np.column_stack((ones, r2[:-1])), r2[1:])
        if opt_result.success:
            self.omega, self.alpha = opt_result.x
            pprint(self.parameters)
        else:
            pprint(opt_result)

    def simulate(self, x0, t):
        x = np.zeros_like(t)
        x[0] = x0
        w, a = self.omega, self.alpha
        for i in range(1, len(t)):
            s = np.sqrt(w + a * x[i-1] ** 2)
            x[i] = np.random.randn() * s
        return x


def main():
    np.random.seed(0)
    a_model = Arch1()
    a_model.omega = 1e-4
    a_model.alpha = 0.3
    dt = 1
    t = np.arange(0, 5000., dt)
    x = a_model.simulate(0, t)
    a_model.fit_parameters(dt, x)
    y = a_model.simulate(0, t)
    plt.plot(t, x.cumsum())
    plt.plot(t, y.cumsum())
    plt.show()


if __name__ == "__main__":
    main()
