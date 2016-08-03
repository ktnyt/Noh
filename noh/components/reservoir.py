import numpy as np
from noh.components import Layer
from noh.activate_functions import sigmoid

class Reservoir(Layer):
    
    def __init__(self, n_visible, n_hidden, activate=sigmoid, bind_prob_W=0.03, bind_prob_W_rec=0.1):
        super(Reservoir, self).__init__(n_visible, n_hidden)
        
        self.W = np.zeros((n_visible, n_hidden))
        self.W_rec = np.zeros((n_hidden, n_hidden))

        self.prev_hid = np.zeros(n_hidden)

        for x1 in xrange(n_visible):
            for x2 in xrange(n_hidden):
                if np.random.rand() < bind_prob_W:
                    self.W[x1][x2] = np.random.rand() - 0.5

        for x1 in xrange(n_hidden):
            for x2 in xrange(n_hidden):
                if np.random.rand() < bind_prob_W_rec:
                    self.W_rec[x1][x2] = np.random.rand() - 0.5

        self.activate = activate

    def __call__(self, data):
        return self.prop_up(data)

    def train(self, data, label, epochs, **kwargs):
        return 0

    def prop_up(self, data):
        data = np.atleast_1d(data)

        "Note: In many case, activate function of the reservoir is recomennded to use tanh."
        self.prev_hid = self.activate(np.dot(data, self.W) + np.dot(self.prev_hid, self.W_rec))

        return self.prev_hid

    def prop_up_sequence(self, dataset):
        hid_list = []
        for data in dataset:
            hid_list.append(self.prop_up(data))
        return np.array(hid_list)

    def prop_down(self, data):
        raise NotImplementationError("Reservoir Class could not prop_down")

    
