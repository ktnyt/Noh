from noh.environment import SupervisedEnvironment

import numpy as np

class SimpleTest(SupervisedEnvironment):

    n_visible = 10
    n_dataset = 10

    x = np.random.randint(0, 2, size=(n_visible, n_dataset))
    y = np.identity(n_dataset)
            
    dataset = (x, y)

    def __init__(self, model):
        super(SimpleTest, self).__init__(model)

    def train(self, epochs=None):
        super(SimpleTest, self).train(epochs=1000)
    
