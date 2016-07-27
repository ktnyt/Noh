from noh.component import Component
from noh.training_functions import gen_sgd_trainer
from noh.activate_functions import sigmoid

import numpy as np

class Layer(Component):
    def __init__(self, n_visible, n_hidden, train_func_generator=None, activate=sigmoid):
        a = 1. / n_visible

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.array(np.random.uniform(low=-a, high=a, size=(n_visible, n_hidden)), 
                          dtype=np.float32)
        self.b_visible = np.zeros(n_visible, dtype=np.float32)
        self.b_hidden = np.zeros(n_hidden, dtype=np.float32)

        if train_func_generator is None:
            self._train = gen_sgd_trainer(self)
        else:
            self._train = train_func_generator(self)

        self.activate = activate
        self.rng = np.random.RandomState(123)
            
    def __call__(self, data, **kwargs):
        return self.prop_up(data)

    def train(self, data, label, epochs, **kwargs):
        for _ in xrange(epochs):
            error = self._train(data, label)
        return error

    def prop_up(self, v):
        return self.activate(np.dot(v, self.W) + self.b_hidden)

    def prop_down(self, h):
        return self.activate(np.dot(h, self.W.T) + self.b_visible)

    def rec(self, v):
        return self.prop_down(self.prop_up(v))

    def get_rec_square_error(self, v):
        rec_v = self.rec(v)
        return 0.5 * np.sum((v - rec_v)**2) / v.shape[0]

    def get_rec_cross_entropy(self, v):
        rec_v = self.rec(v)
        return - np.mean(np.sum(v * np.log(rec_v) + (1 - v) * np.log(1 - rec_v), axis=1))
