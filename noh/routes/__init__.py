from abc import ABCMeta, abstractmethod
from noh import Route

class PropRoute(Route):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class TrainRoute(Route):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, data, label, epoch):
        raise NotImplementedError("`__call__` must be explicitly overridden")
