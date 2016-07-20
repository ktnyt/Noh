import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from noh.component import Component
#from noh.planner import Plan, Planner
from noh.utils import *
from noh.training_functions import gen_layer_default_trainer

import numpy as np

class Layer(Component):
    def __init__(self, n_visible, n_hidden):
        a = 1. / n_visible

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.array(np.random.uniform(low=-a, high=a, size=(n_visible, n_hidden)), dtype=np.float32)
        self.b_visible = np.zeros(n_visible, dtype=np.float32)
        self.b_hidden = np.zeros(n_hidden, dtype=np.float32)

        self._train = gen_layer_default_trainer(self)

    def __call__(self, data, **kwargs):
        return self.prop_up(data)

    def train(self, data, label, **kwargs):
        return self._train(data, label)

    def prop_up(self, v):
        return sigmoid(np.dot(v, self.W) + self.b_hidden)

    def prop_down(self, h):
        return sigmoid(np.dot(h, self.W.T) + self.b_visible)
