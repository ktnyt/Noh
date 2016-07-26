import numpy as np

def gen_empty_trainer():
    def empty_trainer():
        return None
    return empty_trainer


def gen_sgd_trainer(model):
    from noh.utils import p_sig, get_lr_func
    lr_func = get_lr_func(lr_type="const", lr=0.1)
    def sgd_trainer(x, t):

        lr = lr_func()
        y = model(x)
        error = (t - y)
        model.W += lr * np.dot( (p_sig(y) * error).T, x).T
        model.b_hidden += lr * np.dot((p_sig(y) * error).T,
                                      np.ones((model.n_visible, 1)))[0]
            
        input_error = np.dot(error, model.W.T)

        print np.sum(error**2)*0.5

    return sgd_trainer


def gen_linear_regression_trainer(model):
    def linear_regression_trainer(x, t):
        """ Note: This function set parms directory """
        model.W = np.dot(np.linalg.pinv(x), t)
    return linear_regression_trainer
