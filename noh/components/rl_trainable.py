from abc import ABCMeta, abstractmethod

class RLTrainable(object):

    __metaclass__ = ABCMeta

    rl_trainable = 1

    @abstractmethod
    def set_reward(self):
        raise NotImplementedError("`set_reward` must be explicitly overridden")