import numpy as np
import warnings
from noh.components import RBM
from noh.utils import get_lr_func
from noh.activate_functions import sigmoid, p_sig

class PtReservoir(RBM):
    def __init__(self, n_visible, n_hidden, lr_type="hinton_r_div", r_div=None, lr=None):
        super(PtReservoir, self).__init__(n_visible=(n_visible+n_hidden), n_hidden=n_hidden, 
                                          lr_type=lr_type, r_div=r_div, lr=lr)
        self.prev_hid = np.zeros((1, n_hidden))

    def __call__(self, data):
        self.prop_up(data)

    def train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        self.unsupervised_train(data,  k=k, epochs=epochs)

    def prop_up(self, data):

        data = np.atleast_2d(data)
        data = np.c_[data, self.prev_hid]
        self.prev_hid = super(PtReservoir, self).prop_up(data)
        return self.prev_hid

    def prop_up_sequence(self, dataset):
        hid_list = []
        for data in dataset:
            hid_list.append(self.prop_up(data))
        return np.array(hid_list)

    def prop_down(self, data):
        raise NotImplementationError("prop_down is not implemented now.")

    def supervised_train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        super(RBM, self).train(data, label=None, lr=lr, k=k, epochs=epochs)

    def unsupervised_train(self, data, epochs=1000, minibatch_size=1, k=1):
        warnings.warn("ninibatch_size is not 1")
        for i in xrange(epochs):
            error = 0
            loop_max(max((data.shape[0] / minibatch_size), 1))
            for mb_id in xrange(loop_max):
                mdata = data[mb_id * minibatch_size : min((mb_id+1) * minibatch_size, data.shape[0])]
                joint_mdata = np.c_[mdata, self.prev_hid]
                self.CD(data=joint_mdata)
                self.prop_up(mdata)
                error += self.get_rec_cross_entropy(mdata)
            print "epoch: ", i, error / loop_max

    def get_rec_cross_entropy(self, v):
        joint_v = np.c_[v, self.prev_hid]
        h = super(PtReservoir, self).prop_up(joint_v)
        rec_v = super(PtReservoir, self).prop_down(h)
        return - np.mean(np.sum(joint_v * np.log(rec_v) + (1 - joint_v) * np.log(1 - rec_v), axis=1))

    def rec(self, v):
        raise NotImplementationError("prop_down is not implemented now.")

    def prop_down(self, data):
        raise NotImplementationError("prop_down is not implemented now.")

    def get_energy(self, data):
        raise NotImplementationError("get_energy is not implemented now.")

    def gen_sampled_data(self, hidden_rep=None, n_sample=1, sampling_epochs=1):
        raise NotImplementationError("gen_sampled_data is not implemented now.")
