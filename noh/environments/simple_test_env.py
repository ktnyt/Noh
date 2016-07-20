from noh.environment import SupervisedEnvironment

import numpy as np

class SimpleTest(SupervisedEnvironment):

    def __init__(self, model, dataset=None):
        super(SimpleTest, self).__init__(model, dataset)

        n = 10

        print dataset

        if dataset is None:
            x = np.random.randint(0, 2, size=(n, n))
            y = np.identity(n)
            
        self.dataset = (x, y)
        
        for d in self.dataset:
            print d
