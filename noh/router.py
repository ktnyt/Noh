from abc import ABCMeta, abstractmethod
from noh.utils import DotAccessible

class Route(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.components = DotAccessible({})

    def setup(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Router(object):
    def __init__(self, prop, train, **routes):
        self.prop = prop
        self.train = train
        self.routes = routes
        self.routes['prop'] = prop
        self.routes['train'] = train

    def setup(self, components):
        for name in self.routes:
            route = self.routes[name]
            route.setup(components)

    def set_prop(self, name):
        self.prop = self.routes[name]

    def set_train(self, name):
        self.train = self.routes[name]

    def __call__(self, data):
        return self.prop(data)

    def train(self, data, label, epoch):
        return self.train(data, label, epoch)
