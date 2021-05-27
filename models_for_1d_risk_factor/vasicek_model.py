"""Vasicek model, also known as Ornstein-Uhlenbeck process

dx_t = speed * (target - x_t) * dt + volatility * dw_t
solution: x_t - x_0 = a * x_0 + b * 1 + error
A = - (1 - exp(- speed * t))
B = - A * target
Var(error) = volatility^2 * (1 - exp(- 2 * speed * t)) / (2 * speed)
"""
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from time_series_model_template import TimeSeriesModel


class Vasicek(TimeSeriesModel):
    def __init__(self):
        self.drift = None
        self.speed = None
        self.target = None
        self.volatility = None

    def fit_parameters(self, dt, x, method='OLS'):
        methods = {'OLS': self.fit_parameters_ols,
                   'MLE': self.fit_parameters_mle}
        methods[method](dt, x)

    def fit_parameters_ols(self, dt, x):
        x1 = np.vstack((x[:-1], np.ones_like(x[:-1]))).T
        y = x[1:]
        (a, b), (sum_error2,), _, _ = np.linalg.lstsq(x1, y, rcond=-1)
        self.speed = - np.log(a) / dt
        self.target = b / (1 - a)
        self.drift = self.speed * self.target
        self.volatility = np.sqrt(
            sum_error2 / (len(y) - 2) / (1 - a**2) * (2 * self.speed * dt))

    def fit_parameters_mle(self, dt, x):
        pass

    def simulate(self, x0, t):
        x = np.zeros_like(t)
        x[0] = x0
        a, b, s = self.speed, self.target, self.volatility
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dw = np.random.randn() * np.sqrt(dt)
            x[i] = x[i-1] * np.exp(-a * dt)
            x[i] += b * (1-np.exp(- a * dt))
            x[i] += s * np.sqrt(
                (1 - np.exp(- 2 * a * dt)) / (2 * a)
            ) * np.random.normal()
        return x


def main():
    np.random.seed(0)
    v_model = Vasicek()
    v_speed, v_target, v_volatility = .5, -1., 0.1
    dt = 1./250
    t = np.arange(0, 5., dt)
    n_run = 100
    speeds = []
    targets = []
    volatilities = []
    for _ in range(n_run):
        v_model.speed = v_speed
        v_model.target = v_target
        v_model.volatility = v_volatility
        x = v_model.simulate(0, t)
        v_model.fit_parameters(dt, x)
        speeds.append(v_model.parameters['speed'])
        targets.append(v_model.parameters['target'])
        volatilities.append(v_model.parameters['volatility'])
    str_format = '.2f'
    pprint(f"volatility: {np.min(volatilities):.3f} {np.mean(volatilities):.3f} {np.max(volatilities):.3f}")
    pprint(f"speed: {np.min(speeds):.3f} {np.mean(speeds):.3f} {np.max(speeds):.3f}")
    pprint(f"target: {np.min(targets):.3f} {np.mean(targets):.3f} {np.max(targets):.3f}")


if __name__ == "__main__":
    main()
