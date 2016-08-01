from abc import ABCMeta, abstractmethod

from noh.component import Component
from noh.utils import Collection

class PropRule(object):

    __metaclass__ = ABCMeta

    def __init__(self, components, labels=None):
        self.components = Collection(values=components, keys=labels)

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class TrainRule(object):

    __metaclass__ = ABCMeta

    def __init__(self, components, labels=None):
        self.components = Collection(values=components, keys=labels)

    @abstractmethod
    def __call__(self, data, label, epoch):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Planner(object):
    def __init__(self, prop, train, components, labels=None, **Rules):
        self.components = Collection(values=components, keys=labels)
        self.rules = {
            'prop': prop(components, labels),
            'train': train(components, labels)
        }

        for name in Rules:
            Rule = Rules[name]
            self.rules[name] = Rule(components, labels)

        self.prop_rule = self.rules['prop']
        self.train_rule = self.rules['train']

    def set_prop(self, name):
        self.prop_rule = self.rules[name]

    def set_train(self, name):
        self.train_rule = self.rules[name]

    def __call__(self, data):
        return self.prop_rule(data)

    def train(self, data, label, epoch):
        return self.train_rule(data, label, epoch)

class Circuit(Component):
    def __init__(self, planner, components, labels=None):
        self.components = Collection(values=components, keys=labels)
        self.planner = planner(components, labels)

    def __call__(self, data):
        return self.planner(data)

    def train(self, data, label, epochs):
        return self.planner.train(data, label, epochs)
