from abc import ABCMeta
from abc import abstractmethod


class TimeSeriesModel(metaclass=ABCMeta):
    @property
    def parameters(self):
        return self.__dict__

    @abstractmethod
    def fit_parameters(self, t, x):
        pass

    @abstractmethod
    def simulate(self, x0, t):
        pass
