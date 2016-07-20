import numpy as np

from noh.components import Layer
from noh.utils import *

rng = np.random.RandomState(123)

def activate(x):
    sample = rng.binomial(size=x.shape, n=1, p=x)
    return sample

class RBM(Layer):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__(n_visible, n_hidden)

    def __call__(self, data, decode=False):
        return self.decode(data) if decode else self.encode(data)

    def train(self, data, lr=0.01, k=1, epochs=1000):
        for epoch in xrange(epochs):
            error = self.cd_k(data, lr=lr, k=k)
        return error

    def cd_k(self, v, lr=0.01, k=1):
        ph_mean, ph_sample = self.sample_h_given_v(v)

        for step in xrange(k):
            if step == 0:
                nv_mean, nv_sample, nh_mean, nh_sample = self.gibbs_hvh(ph_sample)
            else:
                nv_mean, nv_sample, nh_mean, nh_sample = self.gibbs_hvh(nh_sample)

        self.W += lr * (np.dot(v.T, ph_sample) - np.dot(nv_sample.T, nh_mean))
        self.b_visible += lr * np.mean(v - nv_sample, axis=0)
        self.b_hidden += lr * np.mean(ph_sample - nh_mean, axis=0)

        return self.cross_entropy(v)

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.encode(v0_sample)
        h1_sample = activate(h1_mean)
        return h1_mean, h1_sample

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.decode(h0_sample)
        v1_sample = activate(v1_mean)
        return v1_mean, v1_sample

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def cross_entropy(self, v0):
        h0 = self.encode(v0)
        v1 = self.decode(h0)
        return -np.mean(np.sum(v0 * np.log(v1) + (1 - v0) * np.log(1 - v1), axis=1))

    def encode(self, v):
        return sigmoid(np.dot(v, self.W) + self.b_hidden)

    def decode(self, h):
        return sigmoid(np.dot(h, self.W.T) + self.b_visible)

if __name__ == '__main__':
    rbm = RBM(6, 3)

    x = np.array([[1 ,1 ,1 ,0 ,0, 0],
                  [1 ,0 ,1 ,0 ,0, 0],
                  [1 ,1 ,1 ,0 ,0, 0],
                  [0 ,0 ,1 ,1 ,1, 0],
                  [0 ,0 ,1 ,1 ,0, 0],
                  [0 ,0 ,1 ,1 ,1, 0]])

    print rbm.train(x, lr=0.1, epochs=1000)
    v = np.array([[0, 0, 0, 1, 1, 0],
                  [1, 1, 0, 0, 0, 0]])
    h = rbm.encode(v)
    y = rbm.decode(h)
    print v
    print y
