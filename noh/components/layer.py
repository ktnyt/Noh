from noh.component import Component
from noh.planner import Plan, Planner
from noh.utils import *

import numpy as np

class Layer(Component):
    def __init__(self, n_visible, n_hidden):
        a = 1. / n_visible

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.array(np.random.uniform(low=-a, high=a, size=(n_visible, n_hidden)), dtype=np.float32)
        self.b_visible = np.zeros(n_visible, dtype=np.float32)
        self.b_hidden = np.zeros(n_hidden, dtype=np.float32)

    def __call__(self, x, backward=False):
        if backward:
            return sigmoid(np.dot(x, self.W.T) + self.b_visible)
        return sigmoid(np.dot(x, self.W) + self.b_hidden)

    def train(self, data):
        h = call(data)
        y = call(data, backward=True)
        error = mean_squared_error(y, data)
        return tune(error)

    def tune(self, x, error, lr=0.1):
        y = self(x)
        self.W += lr * np.dot((p_sig(y) * error).T, x).T
        self.b_hidden += lr * np.dot((p_sig(y) * error).T, np.ones((x.shape[0], 1)))[0]

        return np.dot(error, self.W.T)
