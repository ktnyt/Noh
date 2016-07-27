import numpy as np

from noh.components import Layer
from noh.utils import get_lr_func
from noh.activate_functions import sigmoid, p_sig

class RBM(Layer):
    def __init__(self, n_visible, n_hidden, lr_type="hinton_r_div", r_div=None, lr=None):
        super(RBM, self).__init__(n_visible, n_hidden)
        self.lr_type=lr_type
        self.get_lr = get_lr_func(lr_type=lr_type, r_div=r_div, lr=lr)

    def train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        self.unsupervised_train(data,  k=k, epochs=epochs)

    def supervised_train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        super(RBM, self).train(data, label=None, lr=lr, k=k, epochs=epochs)

    def unsupervised_train(self, data, epochs=1000, minibatch_size=100, k=1):
        for i in xrange(epochs):
            for mb_id in xrange(max((data.shape[0] / minibatch_size), 1)):
                mdata = data[mb_id * minibatch_size : min((mb_id+1) * minibatch_size, data.shape[0])]
                self.CD(data=mdata)
            print "epoch: ", i, self.get_rec_cross_entropy(data)

    def CD(self, data):

        h_mean = sigmoid(np.dot(data, self.W) + self.b_hidden)
        h_sample = self.rng.binomial(size=h_mean.shape, n=1, p=h_mean)

        nv_mean = sigmoid(np.dot(h_sample, self.W.T) + self.b_visible)
        nv_sample = self.rng.binomial(size=nv_mean.shape, n=1, p=nv_mean)

        nh_mean = sigmoid(np.dot(nv_sample, self.W) + self.b_hidden)
        nh_sample = self.rng.binomial(size=nh_mean.shape, n=1, p=nh_mean)

        dW = (np.dot(data.T, h_sample) - np.dot(nv_sample.T, nh_mean)) / data.shape[0]
        lr = self.get_lr(weight=self.W, d_weight=dW)
        # print "lr = ", lr

        self.W += lr * dW
        self.b_visible += lr * np.mean(data - nv_sample, axis=0) / data.shape[0]
        self.b_hidden += lr * np.mean(h_sample - nh_mean, axis=0) / data.shape[0]

    def get_energy(self, data):
        """ Return Scala Value """
        hid = self.prop_up(data)
        eng = - np.dot(self.b_visible, data.T).sum(axis=0) - \
            np.dot(np.dot(data, self.W).T, hid).sum(axis=0) - \
            np.dot(hid, self.b_hidden).sum(axis=0)

        return eng.mean()

    def gen_sampled_data(self, hidden_rep=None, n_sample=1, sampling_epochs=1):

        if hidden_rep is None:
            h_mean = np.random.random((n_sample, self.n_hidden))
        else:
            h_mean = hidden_rep
            n_sample = len(h_mean)

        h_sample = self.rng.binomial(size=h_mean.shape, n=1, p=h_mean)
        for _ in xrange(sampling_epochs):
            v_mean = self.prop_down(h_sample)
            v_sample = self.rng.binomial(size=v_mean.shape, n=1, p=v_mean)
            h_mean = self.prop_up(v_sample)
            h_sample = self.rng.binomial(size=h_mean.shape, n=1, p=h_mean)
        dataset = self.prop_down(h_sample)
        return dataset

