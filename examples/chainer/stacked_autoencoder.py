import argparse

import numpy as np
import noh
import noh.components as C

import data

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int, help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100, help='learning minibatch size')

args = parser.parse_args()

batchsize = args.batchsize
epoch = args.epoch
units = args.unit
gpu = args.gpu

stacked_autoencoder = noh.Circuit(
    noh.Wrapper(C.Autoencoder(784, units, gpu=gpu), train=[noh.Binder(epochs=epoch, batchsize=batchsize)]),
    noh.Wrapper(C.Autoencoder(units, units, gpu=gpu), train=[noh.Binder(epochs=epoch, batchsize=batchsize)]),
    noh.Wrapper(C.Perceptron(units, 10, gpu=gpu), train=['labels', noh.Binder(epochs=epoch, batchsize=batchsize)]),
)

# Assume this function loads MNIST data
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N_train = 60000
x_train, x_test = np.split(mnist['data'],   [N_train])
y_train, y_test = np.split(mnist['target'], [N_train])
N_test = y_test.size

errors = stacked_autoencoder.train(x_train, labels=y_train)

print 'Autoencoder 1 Loss: {}'.format(errors[0])
print 'Autoencoder 2 Loss: {}'.format(errors[1])
print 'Perceptron Loss: {} Accuracy: {}'.format(errors[2][0], errors[2][1])
