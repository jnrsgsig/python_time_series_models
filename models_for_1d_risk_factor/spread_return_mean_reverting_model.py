"""Spread-return mean-reverting (SRMR) model

SDE:
    dy_t = (Gamma - A * y_t) * dt + Sigma * dw_t
    y_t = vector(x_t, integral_0^t x_tau d_tau
    Gamma = vector(gamma, 0)
    A = matrix(alpha + beta, alpha * beta,
                     - 1,          0      )
    Sigma = vector(sigma, 0)
    y_0 = vector(x_0, ln(s_0))
Solution:
    y_t = exp(- A * t) * y_0 + (I - exp(- A * t)) * A ** -1 * Gamma + integral_0^t exp(A * (tau - t) * Sigma * dw_tau
    exp(A * t) = exp(beta * t) * matrix(beta * T + exp(Delta * t), (beta ** 2 + beta * Delta) * T,
                                        - T,                       1 - beta * T                   )
    where Delta = alpha - beta, T = (exp(Delta * t) - 1) / Delta
    A ** -1 = matrix(0,                -1,
                     1 / alpha / beta, 1 / alpha + 1 / beta)
    A = P * Lambda * P ** -1
    if alpha != beta, LAMBDA = diag(alpha, beta)
    P = matrix(sin(theta),  -sin(phi),
               -cos(theta), cos(phi) )
    P ** -1 = matrix(cos(phi),   sin(phi),
                     cos(theta), sin(theta)) / sin(theta - phi)
    sin(theta) = alpha / sqrt(1 + alpha ** 2)
    cos(theta) = 1 / sqrt(1 + alpha ** 2)
    sin(phi) = beta / sqrt(1 + beta ** 2)
    cos(phi) = 1 / sqrt(1 + beta ** 2)
    if alpha == beta Lambda = alpha * I + N, N = matrix(0, 1,
                                                        0, 0)
    P = matrix(sin(theta),  sin(phi),
               -cos(theta), cos(phi))
    P ** -1 = matrix(cos(phi),   -sin(phi),
                     sin(theta), sin(theta)) / sin(theta + phi)
       sin(theta) = alpha / sqrt(1 + alpha ** 2)
       cos(theta) = 1 / sqrt(1 + alpha ** 2)
       sin(phi) = cos(theta) ** 3 - sin(theta) ** 2 * sqrt(1 + cos(theta) ** 2)
       cos(phi) = sin(theta) * cos(theta) * (cos(theta) + sqrt(1 + cos(theta) ** 2))
       sin(theta + phi) = cos(theta) ** 2
    t * Lambda = t * (alpha * I + N) = diag(1, 1 / t) * (t * alpha * I + N) * diag(1, t)
"""
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from time_series_model_template import TimeSeriesModel


class SpreadReturnMeanReverting(TimeSeriesModel):
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.sigma = None

    @property
    def volatility(self):
        return self.sigma

    @property
    def drift(self):
        return self.gamma

    @property
    def speed(self):
        return self.alpha + self.beta

    @property
    def intensity(self):
        return self.alpha * self.beta

    @property
    def matrix_A(self):
        return np.array([
            self.speed, self.intensity,
            -1, 0
        ]).reshape([2, 2])

    @property
    def matrix_inv_A(self):
        return np.array([
            0, -1,
            1. / self.intensity, self.speed / self.intensity
        ]).reshape([2, 2])

    def matrix_inv_exp_at(self, t_):
        """exp(A * t_)"""
        _a, _b = self.alpha, self.beta
        e__at, e__bt = np.exp(_a * t_), np.exp(_b * t_)
        a_b = _a - _b
        ab = self.intensity
        if _a == _b:
            return np.array([
                e__at * (1 - _a * t_), - _a ** 2 * t_ * e__at,
                t_ * e__at, - (1 + _a * t_) * e__at
            ])
        return np.array([
            (_a * e__at - _b * e__bt) / a_b, ab * (e__at - e__bt) / a_b,
            - (e__at - e__bt) / a_b, - (_b * e__at - _a * e__bt) / a_b
        ]).reshape([2, 2])

    def one_minus_inv_exp_at_inv_a_gamma(self, t_):
        _a, _b = self.alpha, self.beta
        _e__at, _e__bt = np.exp(- _a * t_), np.exp(- _b * t_)
        if _a == _b:
            _entry_0 = self.gamma * t_ * _e__at
        else:
            _entry_0 = - (_e__at - _e__bt) / (_a - _b)
        if _a == _b == 0:
            _entry_1 = t_ ** 2 / 2
        elif _a == 0:
            _entry_1 = (_e__bt + _b * t_ - 1) / _b ** 2
        elif _b == 0:
            _entry_1 = (_e__at + _a * t_ - 1) / _a ** 2
        elif _a == _b:
            _entry_1 = (1 - _e__at - _a * t_ * _e__at) / _a ** 2
        else:
            _entry_1 = (1 + (_b * _e__at - _a * _e__bt) / (_a - _b)) / self.intensity
        return np.array([_entry_0, _entry_1 * self.gamma])

    def effective_variance(self, t_):
        _a, _b, _s = self.alpha, self.beta, self.sigma
        _at, _bt = _a * t_, _b * t_
        e__at, e__bt = np.exp(- _at), np.exp(- _bt)
        e__2at, e__2bt = e__at ** 2, e__bt ** 2
        var_0, var_1 = None, None
        if np.isclose(_a, 0) and np.isclose(_b, 0):
            var_0 = t_,
            var_1 = t_ ** 3 / 3
        elif np.isclose(_a, 0):
            var_0 = (1 - e__2bt) / 2 / _b
            # var_1 = ((1 - e__2bt) / 2 / _b - 2 / _b * (1 - e__bt) + t_) / _b ** 2
            var_1 = (1 + 2 * _bt - (2 - e__bt) ** 2) / 2 / _b ** 3
        elif np.isclose(_b, 0):
            var_0 = (1 - e__2at) / 2 / _a
            # var_1 = ((1 - e__2at) / 2 / _a - 2 / _a * (1 - e__at) + t_) / _a ** 2
            var_1 = (1 + 2 * _at - (2 - e__at) ** 2) / 2 / _a ** 3
        elif np.isclose(_a + _b, 0):
            sh_2at = np.sinh(2 * _a * t_)
            var_0 = (sh_2at / 2 / _a + t_) / 2
            var_1 = (sh_2at / 2 / _a - t_) / 2 / _a ** 2
        elif np.isclose(_a, _b):
            # var_0 = e__2at * (1 - _at) * t_ / 2 + (1 - e__2at) / _a / 4
            var_0 = (1 - e__2at * ((1 + _at) ** 2 + _at ** 2)) / 4 / _a
            # var_1 = ((1 - e__2at) - 2 * _at * (_at + 1) * e__2at) / _a ** 3 / 4
            var_1 = (1 - e__2at * ((1 - _at) ** 2 + _at ** 2)) / 4 / _a ** 3
        else:
            # var_0 = (_a / 2 * (1 - e__2at) + _b / 2 * (1 - e__2bt) -
            #          2 * self.intensity / self.speed * (1 - e__at * e__bt)) / (_a - _b) ** 2
            var_0 = (1 -
                     ((_b * e__bt - _a * e__at) ** 2 + self.intensity * (e__at - e__bt) ** 2) / (_a - _b) ** 2
                     ) / self.speed / 2
            # var_1 = ((1 - e__2at) / 2 / _a + (1 - e__2bt) / 2 / _b -
            #          2 / self.speed * (1 - e__at * e__bt)) / (_a - _b) ** 2
            var_1 = (1 -
                     ((_a * e__bt - _b * e__at) ** 2 + self.intensity * (e__at - e__bt) ** 2) / (_a - _b) ** 2
                     ) / self.speed / self.intensity / 2
        return self.sigma ** 2 * np.array([var_0, var_1], dtype=float)

    def fit_parameters(self, dt, x, method='OLS'):
        methods = {'OLS': self.fit_parameters_ols,
                   'MLE': self.fit_parameters_mle}
        methods[method](dt, x)

    def fit_parameters_ols(self, dt, x):
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
        y = np.zeros([len(t), 2])
        y[0, 0] = x0
        for i in range(1, len(t)):
            _dt = t[i] - t[i - 1]
            _sigma = np.sqrt(self.effective_variance(_dt))
            y[i] = self.matrix_inv_exp_at(_dt).dot(
                y[i - 1]) + self.one_minus_inv_exp_at_inv_a_gamma(_dt) + np.random.normal(0, _sigma)
        return y[:, 0]


def main():
    np.random.seed(0)
    v_model = SpreadReturnMeanReverting()
    v_alpha, v_beta, v_gamma, v_sigma = 0.0, 0.0, 0.0, 0.03
    dt = 1. / 252
    t = np.arange(0, 5., dt)
    n_run = 100
    speeds = []
    drifts = []
    volatility = []
    for _ in range(n_run):
        v_model.alpha = v_alpha
        v_model.beta = v_beta
        v_model.gamma = v_gamma
        v_model.sigma = v_sigma
        x0 = np.random.normal(1e-2, v_sigma * np.sqrt(dt))
        x = v_model.simulate(x0, t)
        # v_model.fit_parameters(dt, x)
        plt.plot(t, x)
        speeds.append(v_model.speed)
        drifts.append(v_model.drift)
        volatility.append(v_model.volatility)
        # break
    str_format = '.2f'
    pprint(f"volatility: {np.min(volatility):.3f} {np.mean(volatility):.3f} {np.max(volatility):.3f}")
    pprint(f"speed: {np.min(speeds):.3f} {np.mean(speeds):.3f} {np.max(speeds):.3f}")
    pprint(f"drift: {np.min(drifts):.3f} {np.mean(drifts):.3f} {np.max(drifts):.3f}")
    plt.show()


if __name__ == "__main__":
    main()
