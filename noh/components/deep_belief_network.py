import numpy as np

from noh import Circuit
from noh.planner import Plan
from noh.utils import *

from noh.components import RBM, Layer

class Pretrain(Plan):
    def __call__(self, data, lr=0.01, k=1, epochs=1000):
        components = self.components[:-1]
        errors = []

        for component in components:
            error = component.train(data, lr=lr, k=k, epochs=epochs)
            errors.append(error)
            data = component(data)

        return errors

class Finetune(Plan):
    def __call__(self, data, labels, lr=0.1, epochs=1000):
        components = reversed(self.components)

        for epoch in range(epochs):
            error = mean_squared_error(data, labels)
            for component in components:
                error = component.tune(data, error)

        return error

class DeepBeliefNetwork(Circuit):
    def __init__(self, components):
        super(DeepBeliefNetwork, self).__init__(components)
        self.pretrain = Pretrain(components)
        self.finetune = Finetune(components)

    def train(self, data, labels):
        self.pretrain(data, lr=0.1)
        self.finetune(data, labels)

def main():
    x = np.eye(8, dtype=np.float32)
    y = np.array([t.argmax() for t in x]).astype(np.int32)

    layers = [RBM(8, 8), Layer(8, 8)]

    dbn = DeepBeliefNetwork(layers)

    dbn.train(x, y)

    t = dbn(x)

    print t

def hoge():
    

if __name__ == '__main__':
    main()
