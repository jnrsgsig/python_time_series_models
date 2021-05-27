"""GARCH(1, 1) model:

r_t = log(p_t) - log(p_(t-1))
r_t = s_t * e_t
s_t^2 = w + a * r_t^2 + b * s_(t-1)^2
"""
from pprint import pprint
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time_series_model_template import TimeSeriesModel


class Garch11(TimeSeriesModel):
    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
        self.sigma0 = 1

    def fit_parameters(self, dt, x):
        def minus_log_likelihood(w_a_b, var_0=self.sigma0):
            w, a, b = w_a_b
            var_t = var_0
            pi2 = 2 * np.pi
            sum_log_p = 0
            for x_t in x:
                sum_log_p += - np.log(pi2 * var_t) / 2 - x_t ** 2 / var_t / 2
                var_t = w + a * x_t ** 2 + b * var_t
            return - sum_log_p
        # maximum likelihood estimate
        w_a_b_guess = [1e-6, 0.4, 0.4]
        opt_result = minimize(minus_log_likelihood, w_a_b_guess,
                              bounds=[[0, None], [0, 1], [0, 1]])
        if opt_result.success:
            self.omega, self.alpha, self.beta = opt_result.x
            pprint(self.parameters)
        else:
            pprint(opt_result.message)
            pprint(opt_result.x)

    def simulate(self, x0, t):
        x = np.zeros_like(t)
        x[0] = x0
        w, a, b, s = self.omega, self.alpha, self.beta, self.sigma0
        for i in range(1, len(t)):
            s = np.sqrt(w + a * x[i-1] ** 2 + b * s ** 2)
            x[i] = np.random.randn() * s
        return x


def main():
    np.random.seed(0)
    g_model = Garch11()
    g_model.omega = 1e-4
    g_model.alpha = 0.3
    g_model.beta = 0.4
    dt = 1
    t = np.arange(0, 5000., dt)
    x = g_model.simulate(0, t)
    g_model.fit_parameters(dt, x)
    y = g_model.simulate(0, t)
    plt.plot(t, x.cumsum())
    plt.plot(t, y.cumsum())
    plt.show()


if __name__ == "__main__":
    main()
