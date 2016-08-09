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

    def __init__(self, model, render=False):
        super(ReinforcementEnvironment, self).__init__(model)

        if not model.rl_trainable:
            raise ValueError("model should be RL trainable")

        self.render = render
        self.episode_number = 0

    def train(self):
        self.model.train()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError("step must be explicitly overridden")

    @abstractmethod
    def exec_episode(self):
        raise NotImplementedError("exec_episode must be explicitly overridden")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("`reset` must be explicitly overridden")

    @abstractmethod
    def print_stat(self):
        raise NotImplementedError("`print_stat` must be explicitly overridden")


class SupervisedEnvironment(Environment):

    dataset = None
    test_dataset = None

    def __init__(self, model):
        super(SupervisedEnvironment, self).__init__(model)

    def train(self, epochs):
        self.model.train(data=self.dataset[0], label=self.dataset[1], epochs=epochs)

    @classmethod
    def get_dataset(cls):
        return cls.dataset

    @classmethod
    def get_test_dataset(cls):
        return cls.test_dataset


class UnsupervisedEnvironment(Environment):

    dataset = None
    test_dataset = None

    def __init__(self, model):
        super(UnsupervisedEnvironment, self).__init__(model)

    def train(self, epochs):
        self.model.train(data=self.dataset, label=None, epochs=epochs)

    @classmethod
    def get_dataset(cls):
        return cls.dataset

    @classmethod
    def get_test_dataset(cls):
        return cls.test_dataset
