import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

from noh.component import Component

class Perceptron(Component):
    class Model(chainer.Chain):
        def __init__(self, n_input, n_output):
            super(Perceptron.Model, self).__init__(
                transform=F.Linear(n_input, n_output)
            )

        def __call__(self, x):
            y = F.sigmoid(self.transform(x))
            return y

    def __init__(self, n_input, n_output, optimizer=optimizers.Adam, gpu=-1):
        self.model = Perceptron.Model(n_input, n_output)
        self.gpu = gpu
        self.xp = np if gpu < 0 else cuda.cupy

        self.optimizer = optimizer()
        self.optimizer.setup(self.model)

    def __call__(self, data):
        return self.model(chainer.Variable(data)).data

    def train(self, data, labels=None, epochs=1000, batchsize=100):
        if labels is None:
            raise TypeError('Labeled data is mandatory')

        if self.gpu >= 0:
            self.model.to_gpu()

        N = len(data)
        final_loss = 0
        final_accuracy = 0

        for epoch in xrange(epochs):
            perm = np.random.permutation(N)
            sum_loss = 0
            sum_accuracy = 0

            for i in xrange(0, N, batchsize):
                x = chainer.Variable(self.xp.asarray(data[perm[i:i+batchsize]]))
                t = chainer.Variable(self.xp.asarray(labels[perm[i:i+batchsize]]))

                self.optimizer.zero_grads()
                y = self.model(x)
                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)
                loss.backward()
                self.optimizer.update()

                sum_loss += loss.data * batchsize
                sum_accuracy += accuracy.data * batchsize

            final_loss = sum_loss / N
            final_accuracy = sum_accuracy / N

        if self.gpu >= 0:
            self.model.to_cpu()

        return final_loss, final_accuracy

if __name__ == '__main__':
    perceptron = Perceptron(8, 8)

    x = np.eye(8, dtype=np.float32)
    y = np.array(xrange(8), dtype=np.int32)

    print perceptron.train(x, labels=y, epochs=1000, batchsize=8)
