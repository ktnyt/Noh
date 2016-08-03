from noh.environment import ReinforcementEnvironment

class LunarLander_v2(ReinforcementEnvironment):
    def __init__(self):
        pass


    def train(self, epochs):
        for epoch in xrange(epochs):
            self.main_loop()


    def main_loop(self):
        stat = self.get_stat()
        act = self.model(stat)
        self.set_act(act)
        reward = self.get_reward()
        self.model.set_reward()


    def get_stat(self):
        """ Return a vector """
        pass


    def set_act(self, act):
        """ set a vector """
        pass


    def get_reward(self):
        """ Return some scholar value """
        pass