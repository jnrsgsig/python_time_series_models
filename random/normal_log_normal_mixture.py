"""Normal-Log-Normal-Mixture random number generator (NLNM)

Use the same symbols as in the paper
    Normal log-normal mixture: leptokurtosis and skewness, Minxian Yang,
        Applied Economics Letters, 2008, 15, 737-742

NLNM variable := u = epsilon * exp(1 / 2 * eta)
epsilon, eta ~ Normal(0, 1)
corr(epsilon, eta) = rho
generate from iid Normal(0, 1), e1, e2
a1 = rho, a2 = sqrt(1 - rho ^ 2)
epsilon = e1, eta = a1 * e1 + a2 + e2
"""
import numpy as np


class NormalLogNormalMixture:
    def __init__(self):
        self.sigma = None
        self.rho = None
        self.rho_thr = None

    @property
    def c1(self):
        r, s2 = self.rho, self.sigma ** 2
        return .5 * r * np.exp(.125 * s2)

    @property
    def c2(self):
        r2, s2 = self.rho ** 2, self.sigma ** 2
        return np.exp(.5 * s2) * (
                1 + r2 * s2 * (1 - .25 * np.exp(- .25 * s2))
        )

    @property
    def c3(self):
        rs, r2, s2 = self.rho * self.sigma, self.rho ** 2, self.sigma ** 2
        return rs * np.exp(1.125 * s2) * (
                (1 - r2) * (4.5 - 1.5 * np.exp(- 1.5 * s2)) +
                r2 * ((4.5 + 3.375 * s2) - 1.5 * (1 + s2) * np.exp(- .5 * s2) + .25 * s2 * np.exp(- .75 * s2))
        )

    @property
    def c4(self):
        r2, s2 = self.rho ** 2, self.sigma ** 2
        r4, s4 = r2 ** 2, s2 ** 2
        return np.exp(2 * s2) * (
                3 * (1 - r2) ** 2 +
                6 * r2 * (1 - r2) * ((1 + 4 * s2) - 1.5 * s2 * np.exp(- .75 * s2) + .25 * s2 * np.exp(- 1.25 * s2)) +
                r4 * ((3 + 24 * s2 + 16 * s4) - (9 + 6.75 * s2) * np.exp(- .75 * s2) +
                      1.5 * (1 + s2) * s2 * np.exp(- 1.25 * s2) - .1875 * s4 * np.exp(- 1.5 * s2))
        )

    @property
    def kurtosis(self, fisher=False):
        if self.rho < self.rho_thr:
            kurt = 3 * np.exp(self.sigma ** 2)
        else:
            kurt = self.c4 / self.c2 ** 2
        return kurt if not fisher else kurt - 3

    @kurtosis.setter
    def kurtosis(self, pearson_kurtosis):
        self.sigma = np.sqrt(np.log(pearson_kurtosis / 3))

    @property
    def skewness(self):
        if self.rho < self.rho_thr:
            rs, s2 = self.rho * self.sigma, self.sigma ** 2
            return .5 * rs * np.exp(.375 * s2) * (9 - 3 * np.exp(- .5 * s2))
        return self.c3 / self.c2 ** 1.5

    @skewness.setter
    def skewness(self, skew_):
        s = self.sigma
        s2 = s ** 2
        self.rho = skew_ * 2 / s / np.exp(.375 * s2) / (9 - 3 * np.exp(- .5 * s2))


if __name__ == '__main__':
    x = NormalLogNormalMixture()
    x.kurtosis = 1250
    x.skewness = 25
    print(x.rho, x.sigma)
