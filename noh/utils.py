import numpy as np


def get_lr_func(lr_type="const", lr=None, r_div=None):

    if lr_type == "const":
        if lr is None:
            raise ValueError("lr should be decided in case lr_type is \"const\" ")
        def opt_const(**kwargs):
            return lr
        return opt_const

    elif lr_type == "hinton_r_div":
        if r_div is None:
            raise ValueError("r_div should be decided in case lr_type is \"hinton_r_div\" ")
        #def opt_hinton(weight, d_weight):
        def opt_hinton(**kwargs):
            # print "abs weight   : ", np.sum(np.abs(kwargs["weight"]))
            # print "abs d_weight : ", np.sum(np.abs(kwargs["d_weight"]))
            return np.sum(np.abs(kwargs["weight"])) / (np.sum(np.abs(kwargs["d_weight"])) * r_div)
        return opt_hinton
    else:
        raise ValueError("{0} is not defined.".format(lr_type))


def DALP(data, epochs, k=1, H0=1, H1=10, lr_type="hinton_r_div", r_div=None, lr=None):
    
    from noh.components import RBM
    n_visible = data.shape[1]

    print "******* RBM0 *******"

    rbm0 = RBM(n_visible=n_visible, n_hidden=H0, lr_type=lr_type, r_div=r_div, lr=lr)
    rbm0.unsupervised_train(data, k=k, epochs=epochs)
    E0 = rbm0.get_rec_cross_entropy(data)

    print "******* RBM1 *******"

    rbm1 = RBM(n_visible=n_visible, n_hidden=H1, lr_type=lr_type, r_div=r_div, lr=lr)
    rbm1.unsupervised_train(data, k=k, epochs=epochs)
    E1 = rbm1.get_rec_cross_entropy(data)

    print "******* calc *******"

    slope = (E1 - E0) / (H1 - H0)
    H_est = int(- E0 / slope)

    print "H_est = ", H_est

    return H_est

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def mean_squared_error(y, t):
    d = y - t
    d = d.ravel()
    return np.array(d.dot(d) / d.size, dtype=d.dtype)
