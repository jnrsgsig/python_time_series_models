"""AR(1) model: Order 1 auto-regression model

x_1 = c + epsilon_1 + phi * x_0 + theta * epsilon_0
"""
from pprint import pprint

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

from time_series_model_template import TimeSeriesModel


class Arma11(TimeSeriesModel):
    def __init__(self, const=None, ar_l1=None, ma_l1=None, volatility=None):
        self.const = const
        self.ar_l1 = ar_l1
        self.ma_l1 = ma_l1
        self.volatility = volatility

    def fit_parameters(self, dt_, x_, method='OLS'):
        method_func = {
            'OLS': self.fit_parameters_ols,
            'MLE': self.fit_parameters_mle
        }  # '-2' due to two degree of freedom: (1) constant and (2) lagged series
        method_func[method](dt_, x_)

    def fit_parameters_mle(self, dt_, x_):
        def minus_log_likelihood(const_phi_theta_sigma2_epsilon0):
            _c, _phi, _theta, _sigma2, _epsilon = const_phi_theta_sigma2_epsilon0

            def log_p_normal(_z):
                return (- np.log(2 * np.pi * _sigma2)
                        - _z ** 2 / _sigma2
                        ) / 2

            sum_log_p = log_p_normal(_epsilon)
            for i in range(1, len(x_)):
                _epsilon = x_[i] - _c - _phi * x_[i - 1] - _theta * _epsilon
                sum_log_p += log_p_normal(_epsilon)
            return - sum_log_p
        # maximum likelihood estimate
        c_phi_theta_sigma2_e0_guess = [0, 0, 0, 1, 0]
        opt_result = minimize(minus_log_likelihood, c_phi_theta_sigma2_e0_guess,
                              bounds=[[None, None], [-1, 1], [-1, 1], [np.finfo(float).eps, None], [None, None]])
        if opt_result.success:
            self.const, self.ar_l1, self.ma_l1, _sigma2, _e0 = opt_result.x
            self.volatility = np.sqrt(_sigma2 / dt_)
        else:
            pprint(opt_result.message)
            pprint(opt_result.x)

    def fit_parameters_ols(self, dt_, x_):
        x1 = np.vstack((x_[:-1], np.ones_like(x_[:-1]))).T
        y = x_[1:]
        (self.ar_l1, self.const), _, _, _ = np.linalg.lstsq(x1, y, rcond=-1)
        _e = x_[1:] - self.const - self.ar_l1 * x_[:-1]
        var_ = np.var(_e)
        cov_ = (_e[1:] * _e[:-1]).mean() - _e[1:].mean() * _e[:-1].mean()
        if cov_ == 0:
            self.ma_l1 = 0
        else:
            ratio = var_ / cov_
            self.ma_l1 = (ratio + np.sqrt(ratio ** 2 - 4)) / 2
            if np.abs(self.ma_l1) > 1:
                self.ma_l1 = 1. / self.ma_l1
        s2 = var_ / (1 + self.ma_l1 ** 2)
        self.volatility = np.sqrt(s2 / dt_)

    def simulate(self, x0, t_):
        _dt = np.diff(t_).mean()
        _x = np.zeros_like(t)
        _x[0] = x0
        c, phi, theta = self.const, self.ar_l1, self.ma_l1
        _e = np.random.normal(0, self.volatility * np.sqrt(_dt), len(t_))
        _x[1:] = c + _e[1:] + phi * _x[:-1] + theta * _e[:-1]
        return _x


if __name__ == '__main__':
    # np.random.seed(0)
    dt = 1. / 252
    arma_ = Arma11(const=.5, ar_l1=-0.5, ma_l1=.1, volatility=1)
    t = np.arange(0, 30, dt)
    x = arma_.simulate(x0=0, t_=t)
    pprint(arma_.parameters)
    arma_.fit_parameters(dt_=dt, x_=x, method='MLE')
    pprint(arma_.parameters)
    arma_.fit_parameters(dt_=dt, x_=x, method='OLS')
    pprint(arma_.parameters)
