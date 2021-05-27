"""MA(1) model: Order 1 moving average model

x_1 = mu + epsilon_1 + theta * epsilon_0
"""
from pprint import pprint

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from time_series_model_template import TimeSeriesModel


class Ma1(TimeSeriesModel):
    def __init__(self, const=None, ma_l1=None, volatility=None):
        self.const = const
        self.ma_l1 = ma_l1
        self.volatility = volatility

    def fit_parameters(self, dt_, x_):
        self.const = np.mean(x_)
        var_ = np.var(x_)
        cov_ = np.multiply(x_[:-1], x_[1:]).mean() - np.mean(x_[:-1]) * np.mean(x_[1:])
        if cov_ == 0:
            self.ma_l1 = 0
        else:
            ratio = var_ / cov_
            self.ma_l1 = (ratio + np.sqrt(ratio ** 2 - 4)) / 2
            if np.abs(self.ma_l1) > 1:
                self.ma_l1 = 1 / self.ma_l1
        s2 = var_ / (1 + self.ma_l1 ** 2)
        self.volatility = np.sqrt(s2 / dt_)

    def simulate(self, x0, t_):
        _dt = np.diff(t_).mean()
        _x = np.zeros_like(t_, dtype=float)
        _x[0] = x0
        mu, theta = self.const, self.ma_l1
        _e = np.random.normal(0, self.volatility * np.sqrt(_dt), len(t_))
        _x[1:] = mu + _e[1:] + theta * _e[:-1]
        return _x


if __name__ == '__main__':
    # np.random.seed(0)
    dt = 1./252
    ma_ = Ma1(const=1, ma_l1=0.1, volatility=50)
    t = np.arange(0, 5, dt)
    x = ma_.simulate(x0=1, t_=t)
    pprint(ma_.parameters)
    ma_.fit_parameters(dt, x)
    pprint(ma_.parameters)
