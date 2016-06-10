from abc import ABCMeta, abstractmethod

class Component(object):
    """ Callable Component for Circuits with training capability.

    Bind, Circuit, and all component implementations defined in :mod:
    `noh.components` inherit this class.

    :class: `Component`s are the basic building blocks for :class: `Circuits`
    which must provide interfaces to be called and trained. Every component is
    expected to have the :meth: `__call__` method return a meaningful value
    given some sort of input. This is because every :class: `Component` is
    treated as a function which calculates a mapping of a given input.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

    @abstractmethod
    def train(self, data, **kwargs):
        raise NotImplementedError("`train` must be explicitly overridden")
