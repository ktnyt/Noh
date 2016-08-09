from abc import ABCMeta, abstractmethod

from noh.component import Component
from noh.utils import DotAccessible

class PropRule(object):

    __metaclass__ = ABCMeta

    def __init__(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class TrainRule(object):

    __metaclass__ = ABCMeta

    def __init__(self, components):
        self.components = DotAccessible(components)

    @abstractmethod
    def __call__(self, data, label, epoch):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Planner(object):
    def __init__(self, components, PropRules, TrainRules, default_prop_name=None, default_train_name=None):
        self.components = DotAccessible(components)

        self.prop_rules = {}
        for name in PropRules:
            self.prop_rules[name] = PropRules[name](components)
        if default_prop_name is None:
            self.default_prop_rule = self.prop_rules.values()[0]
        else:
            self.default_prop_rule = self.prop_rules[default_prop_name]

        self.train_rules = {}
        for name in TrainRules:
            self.train_rules[name] = TrainRules[name](components)
        if default_train_name is None:
            self.default_train_rule = self.train_rules.values()[0]
        else:
            self.default_train_rule = self.train_rules[default_train_name]

    def set_prop(self, name):
        self.default_prop_rule = self.prop_rules[name]

    def set_train(self, name):
        self.default_train_rule = self.train_rules[name]

    def __call__(self, data):
        return self.default_prop_rule(data)

    def train(self, data, label=None, reward=None, epochs=1):
        return self.default_train_rule(data, label, reward, epochs)

class Circuit(Component):
    def __init__(self, planner, rl_trainable=False, **components):
        super(Circuit, self).__init__(rl_trainable=rl_trainable)
        self.components = DotAccessible(components)
        self.planner = planner(components)


    def __call__(self, data):
        return self.planner(data)

    def train(self, data, label, epochs):
        return self.planner.train(data, label, epochs)
