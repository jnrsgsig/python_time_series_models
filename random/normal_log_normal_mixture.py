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
import scipy.stats as sp_ss
from scipy.optimize import minimize


class NormalLogNormalMixture:
    def __init__(self):
        self.sigma = None
        self.rho = None
        self.scale = None
        self.intercept = None

    @classmethod
    def nlnm_log_c1_per_rho(cls, sigma_rho_):
        s, r = sigma_rho_
        return np.log(.5 * s) + .125 * s ** 2

    @classmethod
    def nlnm_c1(cls, sigma_rho_):
        _, r = sigma_rho_
        return r * np.exp(cls.nlnm_log_c1_per_rho(sigma_rho_))
        # s, r = sigma_rho_
        # return .5 * r * np.exp(.125 * s ** 2)

    @property
    def c1(self):
        return self.nlnm_c1([self.sigma, self.rho])

    @classmethod
    def nlnm_log_c2(cls, sigma_rho_):
        s, r = sigma_rho_
        r2, s2 = r ** 2, s ** 2
        log_term1 = np.log(1 + r2 * s2 * (1 - .25 * np.exp(- .25 * s2)))
        return .5 * s2 + log_term1

    @classmethod
    def nlnm_c2(cls, sigma_rho_):
        return np.exp(cls.nlnm_log_c2(sigma_rho_))

    @property
    def c2(self):
        return self.nlnm_c2([self.sigma, self.rho])

    @classmethod
    def nlnm_log_c3_per_rho_sigma(cls, sigma_rho_):
        s, r = sigma_rho_
        rs, r2, s2 = r * s, r ** 2, s ** 2
        log_term1 = np.log(
            (1 - r2) * (4.5 - 1.5 * np.exp(- .5 * s2)) +
            r2 * ((4.5 + 3.375 * s2) - 1.5 * (1 + s2) * np.exp(- .5 * s2) + .25 * s2 * np.exp(- .75 * s2))
        )
        return 1.125 * s2 + log_term1

    @classmethod
    def nlnm_c3(cls, sigma_rho_):
        s, r = sigma_rho_
        return r * s * np.exp(cls.nlnm_log_c3_per_rho_sigma(sigma_rho_))

    @property
    def c3(self):
        return self.nlnm_c3([self.sigma, self.rho])

    @classmethod
    def nlnm_log_c4(cls, sigma_rho_):
        s, r = sigma_rho_
        r2, s2 = r ** 2, s ** 2
        r4, s4 = r2 ** 2, s2 ** 2
        log_term1 = np.log(
            3 * (1 - r2) ** 2 +
            6 * r2 * (1 - r2) * ((1 + 4 * s2) - 1.5 * s2 * np.exp(- .75 * s2) + .25 * s2 * np.exp(- 1.25 * s2)) +
            r4 * ((3 + 24 * s2 + 16 * s4) - (9 + 6.75 * s2) * np.exp(- .75 * s2) +
                  1.5 * (1 + s2) * s2 * np.exp(- 1.25 * s2) - .1875 * s4 * np.exp(- 1.5 * s2))
        )
        return 2 * s2 + log_term1

    @classmethod
    def nlnm_c4(cls, sigma_rho_):
        return np.exp(cls.nlnm_log_c4(sigma_rho_))

    @property
    def c4(self):
        return self.nlnm_c4([self.sigma, self.rho])

    @classmethod
    def nlnm_log_kurt(cls, sigma_rho_):
        return cls.nlnm_log_c4(sigma_rho_) - cls.nlnm_log_c2(sigma_rho_) * 2

    @classmethod
    def nlnm_kurt(cls, sigma_rho_, fisher=False):
        kurt = np.exp(cls.nlnm_log_kurt(sigma_rho_))
        return kurt - 3 if fisher else kurt

    @classmethod
    def nlnm_fisher_kurt(cls, sigma_rho_):
        return cls.nlnm_kurt(sigma_rho_, fisher=True)

    @classmethod
    def nlnm_pearson_kurt(cls, sigma_rho_):
        return cls.nlnm_kurt(sigma_rho_, fisher=False)

    @property
    def kurt_excess(self):
        # kurt = 3 * np.exp(s ** 2)
        return self.nlnm_fisher_kurt([self.sigma, self.rho])

    @classmethod
    def nlnm_skew(cls, sigma_rho_):
        s, r = sigma_rho_
        log_skew_per_rho = cls.nlnm_log_c3_per_rho_sigma(sigma_rho_) - cls.nlnm_log_c2(sigma_rho_) * 1.5
        return s * r * np.exp(log_skew_per_rho)

    @property
    def skew(self):
        # return .5 * rs * np.exp(.375 * s2) * (9 - 3 * np.exp(- .5 * s2))
        return self.nlnm_skew([self.sigma, self.rho])

    @property
    def var(self):
        if self.scale:
            return self.scale ** 2 * self.c2
        return self.c2

    @property
    def mean(self):
        k = self.scale if self.scale else 1
        b = self.intercept if self.intercept else 0
        return k * self.c1 + b

    @classmethod
    def learn_sigma_rho_from(cls, skew_, kurt_excess_):
        kurt = kurt_excess_ + 3
        s0 = np.sqrt(np.log(kurt / 3))
        r0 = skew_ * 2 / s0 * np.exp(- .375 * s0 ** 2) / (9 - 3 * np.exp(- .5 * s0 ** 2))
        if r0 > 1:
            r0 = 1
        if r0 < -1:
            r0 = -1
        print(s0, r0)
        f_skew = cls.nlnm_skew
        f_log_kurt = cls.nlnm_log_kurt

        def err2(sigma_rho_shift):
            s, r, skew2_shift = sigma_rho_shift
            sigma_rho_ = [s, r]
            # skew2_shift = np.exp(7)
            err2_skew = (np.log(f_skew(sigma_rho_)**2 + skew2_shift) / 3 - np.log(skew_**2 + skew2_shift) / 3)**2
            err2_kurt = (f_log_kurt(sigma_rho_) / 2 - np.log(kurt) / 2)**2
            return err2_skew + err2_kurt

        return minimize(err2, np.array([s0, r0, 1]), bounds=[[0, None], [-1, 1], [0, None]])

    @classmethod
    def search_rho_to_fit_kurtosis(cls, kurt_excess_):
        kurt = kurt_excess_ + 3
        s0 = np.sqrt(np.log(kurt / 3))
        skew_ = []
        for _r in np.linspace(0, 1, 1001):
            f_log_kurt = lambda x_: cls.nlnm_log_kurt([x_, _r])
            err2 = lambda sigma_rho_: (f_log_kurt(sigma_rho_) - np.log(kurt)) ** 2
            _s, = minimize(err2, s0).x
            skew_.append(cls.nlnm_skew([_s, _r]))
            # kurt_ = cls.nlnm_fisher_kurt([_s, _r])
        print(max(skew_))

    def learn_from_sample(self, sample_array):
        mean_ = np.mean(sample_array)
        var_ = np.var(sample_array)
        skew_ = sp_ss.skew(sample_array)
        kurt_excess_ = sp_ss.kurtosis(sample_array)
        opt_res_ = self.learn_sigma_rho_from(skew_, kurt_excess_)
        if not opt_res_.success:
            raise ValueError(opt_res_.message)
        self.sigma, self.rho, _ = opt_res_.x
        self.scale = np.sqrt(var_ / self.c2)
        self.intercept = mean_ - self.c1 * self.scale

    def __call__(self, *args, **kwargs):
        b = self.intercept if self.intercept else 0
        k = self.scale if self.scale else 1
        r, s = self.rho, self.sigma
        if not args and not kwargs:
            eps = np.random.normal(0, 1)
            eta = np.random.normal(0, s)
            eta = r * eps + np.sqrt(1 - r**2) * eta
            return np.exp(.5 * eta) * eps * k + b
        n, = args
        eps = np.random.normal(0, 1, n)
        eta = np.random.normal(0, 1, n)
        eta = (r * eps + np.sqrt(1 - r ** 2) * eta) * s
        return np.exp(.5 * eta) * eps * k + b


if __name__ == '__main__':
    x = NormalLogNormalMixture()
    opt_res = x.learn_sigma_rho_from(skew_=7.58, kurt_excess_=137.2)
    print(opt_res.success, opt_res.x)
    x.sigma, x.rho, _ = opt_res.x
    print(x.mean, x.var, x.skew, x.kurt_excess)
    # x.search_rho_to_fit_kurtosis(kurt_excess_=137.2)
    print(x.scale, x.intercept, x.sigma, x.rho)
    y = x(10**7) * 10 - 10
    print(np.mean(y), np.var(y), sp_ss.skew(y), sp_ss.kurtosis(y))
    x.learn_from_sample(y)
    print(x.mean, x.var, x.skew, x.kurt_excess)
