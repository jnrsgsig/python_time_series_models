"""Cox-Ingersoll-Ross (CIR) model

dx_t = (drift - speed * x_t) * dt + volatility * sqrt(x_t) * dw_t
d[sqrt(x_t)] = ((drift / 2 - volatility ** 2 / 4) / sqrt(x_t) - speed / 2 * sqrt(x_t)) * dt + volatility / 2 * dw_t
"""
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from time_series_model_template import TimeSeriesModel


class CoxIngersollRoss(TimeSeriesModel):
    def __init__(self):
        self.drift = None
        self.speed = None
        self.volatility = None

    def fit_parameters(self, dt, x, method='OLS'):
        methods = {'OLS': self.fit_parameters_ols,
                   'MLE': self.fit_parameters_mle}
        methods[method](dt, x)

    def fit_parameters_ols(self, dt, x):
        sqrt_x = np.sqrt(x)
        y = np.diff(x) / sqrt_x[:-1]
        xx = np.vstack((sqrt_x[:-1], 1. / sqrt_x[:-1])).T
        (a, b), (sum_error2,), _, _ = np.linalg.lstsq(xx, y, rcond=-1)
        self.speed = - a / dt
        self.drift = b / dt
        sigma2 = sum_error2 / (len(y) - 2) / dt
        self.volatility = np.sqrt(sigma2)

    def fit_parameters_ols_2(self, dt, x):
        sqrt_x = np.sqrt(x)
        xx = np.vstack((sqrt_x[:-1], 1 / sqrt_x[:-1])).T
        y = sqrt_x[1:]
        (a, b), (sum_error2,), _, _ = np.linalg.lstsq(xx, y, rcond=-1)
        self.speed = 2 * (1 - a) / dt
        sigma2 = sum_error2 / (len(y) - 2) * 4 / dt
        self.volatility = np.sqrt(sigma2)
        self.drift = 2 * (b / dt + sigma2 / 4)

    def fit_parameters_mle(self, dt, x):
        pass

    def simulate(self, x0, t):
        x = np.zeros_like(t)
        # sqrt_x = np.zeros_like(t)
        x[0] = x0
        # sqrt_x[0] = np.sqrt(x0)
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dw = np.random.normal(0, 1) * np.sqrt(dt)
            # sqrt_x[i] = sqrt_x[i - 1] + (
            #         (self.drift / 2 - self.volatility ** 2 / 4) / sqrt_x[i - 1] - self.speed / 2 * sqrt_x[i - 1]
            # ) * dt + self.volatility / 2 * dw
            x[i] = x[i - 1] + (self.drift - self.speed * x[i - 1]) * dt + self.volatility * np.sqrt(x[i - 1]) * dw
        # x = sqrt_x
        return x


def main():
    np.random.seed(0)
    v_model = CoxIngersollRoss()
    v_speed, v_drift, v_volatility = .1, .01, 0.03
    dt = 1./252
    t = np.arange(0, 30., dt)
    n_run = 100
    speeds = []
    drifts = []
    volatility = []
    for _ in range(n_run):
        v_model.speed = v_speed
        v_model.drift = v_drift
        v_model.volatility = v_volatility
        x0 = np.random.normal(1e-2, v_volatility * np.sqrt(dt))
        x = v_model.simulate(x0, t)
        v_model.fit_parameters(dt, x)
        plt.plot(t, x)
        speeds.append(v_model.parameters['speed'])
        drifts.append(v_model.parameters['drift'])
        volatility.append(v_model.parameters['volatility'])
        # break
    str_format = '.2f'
    pprint(f"volatility: {np.min(volatility):.3f} {np.mean(volatility):.3f} {np.max(volatility):.3f}")
    pprint(f"speed: {np.min(speeds):.3f} {np.mean(speeds):.3f} {np.max(speeds):.3f}")
    pprint(f"drift: {np.min(drifts):.3f} {np.mean(drifts):.3f} {np.max(drifts):.3f}")
    plt.show()


if __name__ == "__main__":
    main()
