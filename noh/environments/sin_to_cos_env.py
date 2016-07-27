from noh.environment import SupervisedEnvironment

import numpy as np

class Sin2CosTest(SupervisedEnvironment):

    x_list = np.array([float(x) for x in xrange(360)])
    x = np.array([[d] for d in np.sin(x_list * np.pi / 180. )])
    y = np.array([[d] for d in np.cos(x_list*2 * np.pi / 180. )])
            
    dataset = (x, y)

    def __init__(self, model):
        super(Sin2CosTest, self).__init__(model)
	
    def train(self, epochs):
        super(Sin2CosTest, self).train(epochs=epochs)
    
    def unsupervised_train(self, epochs):
        self.model.unsupervised_train(self.dataset[0], epochs)
