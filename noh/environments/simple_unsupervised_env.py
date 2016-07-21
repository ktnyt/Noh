from noh.environment import UnsupervisedEnvironment

import numpy as np

class SimpleUnsupervisedTest(UnsupervisedEnvironment):

    def __init__(self, model):
        super(SimpleUnsupervisedTest, self).__init__(model)

        self.dataset = np.array([[1, 1, 1, 0, 0, 0],
                                 [1, 0, 1, 0, 0, 0],
                                 [1, 1, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 1, 0],
                                 [0, 0, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 1, 0]])

        self.test_dataset = np.array([[0, 0, 0, 1, 1, 0],
                                      [1, 1, 0, 0, 0, 0]])




