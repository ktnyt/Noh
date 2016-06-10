import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

from noh.component import Component

class Autoencoder(Component):
    class Model(chainer.Chain):
        def __init__(self, n_input, n_output):
            super(Autoencoder.Model, self).__init__(
                encode=F.Linear(n_input, n_output),
                decode=F.Linear(n_output, n_input),
            )

        def __call__(self, x, train=True):
            h = F.sigmoid(self.encode(F.dropout(x, train=train)))
            y = F.sigmoid(self.decode(F.dropout(h, train=train)))
            return y

    def __init__(self, n_input, n_output, optimizer=optimizers.Adam, gpu=-1):
        self.model = Autoencoder.Model(n_input, n_output)
        self.gpu = gpu
        self.xp = np if gpu < 0 else cuda.cupy
        self.optimizer = optimizer()
        self.optimizer.setup(self.model)

    def __call__(self, data, decode=False):
        if decode:
            return self.decode(data)
        return self.encode(data)

    def train(self, data, epochs=1000, batchsize=100, dropout=True):
        if self.gpu >= 0:
            self.model.to_gpu()

        N = len(data)
        final_loss = 0

        for epoch in xrange(epochs):
            perm = np.random.permutation(N)
            sum_loss = 0

            for i in xrange(0, N, batchsize):
                x = chainer.Variable(self.xp.asarray(data[perm[i:i+batchsize]]))
                t = chainer.Variable(self.xp.asarray(data[perm[i:i+batchsize]]))

                self.optimizer.zero_grads()
                y = self.model(x, train=dropout)
                loss = F.mean_squared_error(y, t)
                loss.backward()
                self.optimizer.update()

                sum_loss += loss.data * batchsize

            final_loss = sum_loss / N

        if self.gpu >= 0:
            self.model.to_cpu()

        return final_loss

    def encode(self, data):
        return F.sigmoid(self.model.encode(chainer.Variable(data))).data

    def decode(self, data):
        return F.sigmoid(self.model.decode(chainer.Variable(data))).data


if __name__ == '__main__':
    autoencoder = Autoencoder(8, 3)

    x = np.eye(8, dtype=np.float32)

    print autoencoder.train(x, epochs=10000, batchsize=8, dropout=False)
    h = autoencoder.encode(x)
    y = autoencoder.decode(h)
    print [t.argmax() for t in x]
    print [t.argmax() for t in y]
