from abc import ABCMeta, abstractmethod


class Environment(object):

    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self):
        raise NotImplementedError("`train` must be explicitly overridden")


class ReinforcementEnvironment(Environment):

    __metaclass__ = ABCMeta

    def __init__(self, model):
        if not model.RL_trainable:
            raise ValueError("model should be RL trainable")
        super(ReinforcementEnvironment, self).__init__(model)

    def train(self):
        stat = self.get_stat()
        act = self.model(stat)
        self.set_act(act)
        reward = self.get_reward()
        self.model.set_reward()

    @abstractmethod
    def get_stat(self):
        """ Return a vector """
        raise NotImplementedError("`get_stat` must be explicitly overridden")

    @abstractmethod
    def set_act(self, act):
        """ set a vector """
        raise NotImplementedError("`set_act` must be explicitly overridden")

    @abstractmethod
    def get_reward(self):
        """ Return some scholar value """
        raise NotImplementedError("`get_stat` must be explicitly overridden")


class SupervisedEnvironment(Environment):

    def __init__(self, model, dataset=None):
        super(SupervisedEnvironment, self).__init__(model)
        self.dataset = dataset

    def train(self, epochs=1000, lr=0.1):
        for _ in xrange(epochs):
            self.model.train(self.dataset[0], self.dataset[1])

    def get_dataset(self):
        return self.dataset


class UnsupervisedEnvironment(Environment):

    def __init__(self, model, dataset=None):
        super(UnsupervisedEnvironment, self).__init__(model)
        self.dataset = dataset

    def train(self, epochs=100, lr=0.1):
        for _ in xrange(epochs):
            self.model.train(self.dataset)

    def get_dataset(self):
        return self.dataset
