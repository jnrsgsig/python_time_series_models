"""AR(1) model: Order 1 auto-regression model

x_1 = c + epsilon_1 + phi * x_0
"""
from pprint import pprint

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

from time_series_model_template import TimeSeriesModel


class Ar1(TimeSeriesModel):
    def __init__(self, const=None, ar_l1=None, volatility=None):
        self.const = const
        self.ar_l1 = ar_l1
        self.volatility = volatility

    def fit_parameters(self, dt_, x_, method='OLS'):
        dof = {'OLS': 2, 'MLE': 0}  # '-2' due to two degree of freedom: (1) constant and (2) lagged series
        x1 = np.vstack((x_[:-1], np.ones_like(x_[:-1]))).T
        y = x_[1:]
        (self.ar_l1, self.const), (sum_error2,), _, _ = np.linalg.lstsq(x1, y, rcond=-1)
        self.volatility = np.sqrt(
            sum_error2 / (len(y) - dof[method]) / dt_)

    def fit_parameters_mle(self, dt_, x_):
        def minus_log_likelihood(c_phi_sigma2):
            c, phi, _sigma2 = c_phi_sigma2
            pi2 = 2 * np.pi
            sum_log_p = 0
            for x_0, x_1 in zip(x_[:-1], x_[1:]):
                sum_log_p += - np.log(pi2 * _sigma2) / 2 - (x_1 - c - phi * x_0) ** 2 / _sigma2 / 2
            return - sum_log_p
        # maximum likelihood estimate
        c_phi_sigma2_guess = [1e-6, 0.4, 0.4]
        opt_result = minimize(minus_log_likelihood, c_phi_sigma2_guess,
                              bounds=[[None, None], [-1, 1], [1e-14, None]])
        if opt_result.success:
            self.const, self.ar_l1, sigma2 = opt_result.x
            self.volatility = np.sqrt(sigma2 / dt)
            pprint(self.parameters)
        else:
            pprint(opt_result.message)
            pprint(opt_result.x)

    def simulate(self, x0, t_):
        _dt = np.diff(t_).mean()
        _x = np.zeros_like(t)
        _x[0] = x0
        c, phi, sigma = self.const, self.ar_l1, self.volatility * np.sqrt(_dt)
        for i in range(1, len(t)):
            _x[i] = c + phi * _x[i - 1] + np.random.normal(0, sigma)
        return _x


if __name__ == '__main__':
    np.random.seed(0)
    n_sample = 750
    dt = 1./252
    ar_ = Ar1(const=1, ar_l1=0.3, volatility=0.2)
    t = np.arange(0, 30, dt)
    x = ar_.simulate(x0=0, t_=t)
    pprint(ar_.parameters)
    ar_.fit_parameters(dt_=dt, x_=x)
    pprint(ar_.parameters)
    ar_.fit_parameters(dt_=dt, x_=x, method='MLE')
    pprint(ar_.parameters)
    mod_ = ARIMA(x, order=(1, 0, 0))
    res_ = mod_.fit()
    ar_params = res_.params
    print(ar_params)
    print(f'{ar_.volatility ** 2 * dt : e}')
