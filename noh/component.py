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
        self.params = {}

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

    @abstractmethod
    def train(self, data, label, epochs, **kwargs):
        raise NotImplementedError("`train` must be explicitly overridden")

    def save_params(self):

        import sys, os
        import numpy as np

        class_name = self.__class__.__name__
        exec_name = os.path.splitext(sys.argv[0])[0]

        cwd = os.getcwd()
        dir_name = cwd + "/" + exec_name + "_save/"
        print dir_name
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for parm in self.params:
            print parm
            file_name =class_name+"_"+parm+".npy"
            np.save(dir_name + file_name, self.params[parm]())

    def reset(self):
        pass
