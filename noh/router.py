from abc import ABCMeta, abstractmethod
from noh.utils import DotAccessible

class Route(object):

    __metaclass__ = ABCMeta

    def __init__(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, data, *args, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Router(object):
    def __init__(self, default, **Routes):
        self.Routes = Routes
        self.Routes['default'] = default
        self.routes = {}

    def setup(self, components):
        for name in self.Routes:
            Route = self.Routes[name]
            self.routes[name] = Route(components)
        self.route = self.routes['default']

    def set(self, name):
        self.route = self.routes[name]

    def __call__(self, data):
        return self.route(data)
