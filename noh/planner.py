from abc import ABCMeta, abstractmethod
from noh.utils import DotAccessible

class Plan(object):

    __metaclass__ = ABCMeta

    def __init__(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, data, label, epoch, *args, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Planner(object):
    def __init__(self, default, **Plans):
        self.Plans = Plans
        self.Plans['default'] = default
        self.plans = {}

    def setup(self, components):
        for name in self.Plans:
            Plan = self.Plans[name]
            self.plans[name] = Plan(components)
        self.plan = self.plans['default']

    def set(self, name):
        self.plan = self.plans[name]

    def __call__(self, data, label, epoch):
        return self.plan(data, label, epoch)
