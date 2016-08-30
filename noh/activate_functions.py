
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

def relu(x):
    return x * (x > 0)

def p_relu(x):
    return 1. * (x > 0)
