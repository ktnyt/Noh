import numpy as np

from noh.components import Layer
from noh.utils import sigmoid, p_sig, get_lr_func

rng = np.random.RandomState(123)

class RBM(Layer):
    def __init__(self, n_visible, n_hidden, lr_type="hinton_r_div", r_div=None, lr=None):
        super(RBM, self).__init__(n_visible, n_hidden)
        print self.W
        self.lr_type=lr_type
        self.get_lr = get_lr_func(lr_type=lr_type, r_div=r_div, lr=lr)

    def train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        self.unsupervised_train(data,  k=k, epochs=epochs)

    def supervised_train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        super(RBM, self).train(data, label=None, lr=lr, k=k, epochs=epochs)

    def unsupervised_train(self, data, k=1, epochs=1000):
        for i in xrange(epochs):
            self.CD(data=data)
            print "epoch: ", i, self.get_rec_crossentropy(data)

    def CD(self, data):

        h_mean = sigmoid(np.dot(data, self.W) + self.b_hidden)
        h_sample = rng.binomial(size=h_mean.shape, n=1, p=h_mean)

        nv_mean = sigmoid(np.dot(h_sample, self.W.T) + self.b_visible)
        nv_sample = rng.binomial(size=nv_mean.shape, n=1, p=nv_mean)

        nh_mean = sigmoid(np.dot(nv_sample, self.W) + self.b_hidden)
        nh_sample = rng.binomial(size=nh_mean.shape, n=1, p=nh_mean)

        dW = (np.dot(data.T, h_sample) - np.dot(nv_sample.T, nh_mean))
        lr = self.get_lr(weight=self.W, d_weight=dW)
        print "lr = ", lr

        self.W += lr * dW
        self.b_visible += lr * np.mean(data - nv_sample, axis=0)
        self.b_hidden += lr * np.mean(h_sample - nh_mean, axis=0)

    def get_energy(self, data):
        """ Return Scala Value """
        hid = self.prop_up(data)
        eng = - np.dot(self.b_visible, data.T).sum(axis=0) - \
            np.dot(np.dot(data, self.W).T, hid).sum(axis=0) - \
            np.dot(hid, self.b_hidden).sum(axis=0)

        return eng.mean()

