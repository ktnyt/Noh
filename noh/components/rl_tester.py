from noh import Component
import numpy as np

class RLTester(Component):

    def __init__(self, n_stat, n_act, n_reward=1):
        super(RLTester, self).__init__(RL_trainable=True)
        self.n_stat = n_stat
        self.n_act = n_act
        self.n_reward = n_reward

    def __call__(self, data):
        act = np.random.randint(0, self.n_stat)
        return act

    def train(self, data, label, epochs):
        pass
