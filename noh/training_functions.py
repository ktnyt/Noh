import numpy as np


def gen_empty_trainer():
    def empty_trainer():
        return None
    return empty_trainer


def gen_layer_default_trainer(model):
    from noh.utils import p_sig
    def layer_default_trainer(x, t):

        lr = 0.1
        y = model(x)
        error = (t - y)
        model.W += lr * np.dot( (p_sig(y) * error).T, x).T
        model.b_hidden += lr * np.dot((p_sig(y) * error).T,
                                     np.ones((model.n_visible, 1)))[0]

        input_error = np.dot(error, model.W.T)

        print np.sum(error**2)*0.5

    return layer_default_trainer
