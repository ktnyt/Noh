from noh import Component
import cPickle as pickle
from noh.activate_functions import sigmoid
import numpy as np


class RLNN(Component):
    def __init__(self, n_visible, n_hidden, n_output, lr=1e-4, gamma=0.99, decay_rate=0.99, resume=False):
        # hyperparameters
        self.n_visible = n_visible
        self.n_hidden = n_hidden  # number of hidden layer neurons
        self.n_output = n_output
        self.learning_rate = lr
        self.gamma = gamma  # discount factor for reward
        self.decay_rate = decay_rate  # decay factor for RMSProp leaky sum of grad^2
        # resume = False # resume from previous checkpoint?
        resume = resume
        if resume:
            self.parms = pickle.load(open('save.p', 'rb'))
        else:
            self.parms = {}
            self.parms['W1'] = np.random.randn(self.n_hidden, self.n_visible) / np.sqrt(self.n_visible)  # "Xavier" initialization
            self.parms['W2'] = np.random.randn(self.n_output, self.n_hidden) / np.sqrt(self.n_hidden)
        self.grad_buffer = {k: np.zeros_like(v) for k, v in
                       self.parms.iteritems()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.parms.iteritems()}  # rmsprop memory

    def __call__(self, stat):
        return self.policy_forward(stat)

    def train(self):
        for k, v in self.parms.iteritems():
            g = self.grad_buffer[k]  # gradient
            self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g ** 2
            self.parms[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            self.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

    def policy_forward(self, x):
        h = np.dot(self.parms['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.parms['W2'], h)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, epx, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """

        #dW2 = np.dot(eph.T, epdlogp).ravel()
        dW2 = np.dot(eph.T, epdlogp).T
        #dh = np.multiply.outer(epdlogp, self.parms['W2'])
        dh = np.dot(epdlogp, self.parms['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        grad = {'W1': dW1, 'W2': dW2}
        for k in self.parms:
            self.grad_buffer[k] += grad[k]  # accumulate grad over batch
        return grad

    def save(self):
        pickle.dump(self.parms, open('save.p', 'wb'))