import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def p_sig(op):
    return op * (1. - op)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def mean_squared_error(y, t):
    d = y - t
    d = d.ravel()
    return np.array(d.dot(d) / d.size, dtype=d.dtype)
