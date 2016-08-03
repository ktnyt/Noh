
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def p_sig(x):
    return x * (1. - x)

def linear(x):
    return x

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e)
