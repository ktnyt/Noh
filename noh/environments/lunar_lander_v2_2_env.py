import gym

from noh.environment import ReinforcementEnvironment


class LunarLander_v2(ReinforcementEnvironment):

    n_stat = 8
    n_act = 4
    episode_size = 1000

    @classmethod
    def gen_reward_reset_checker(cls):
        return lambda x: True if x == 100 or x == -100 else False

    def __init__(self, model, render=False):
        super(LunarLander_v2, self).__init__(model, render)
        self.env = gym.make("LunarLander-v2")
        self.batch_size = 10
        self.action = None
        self.frame = 0

    def step(self, action):
        return self.env.step(action)

    def exec_episode(self):
        episode_reward = 0

        observation = self.env.reset()
        done = False
        reward = 0
        for frame in xrange(self.__class__.episode_size):
            if self.render: self.print_stat()
            action = self.model(observation)
            observation, reward, done, info = self.env.step(action)
            self.model.set_reward(reward)
            episode_reward += reward
            if done: break

        if not done:
            print "--- over ---"
            self.model.set_reward(-100)
            episode_reward += (-reward - 100)


        self.model.train()
        self.episode_number += 1
        print ('ep %d: game finished, reward: %f' %
               (self.episode_number, episode_reward))

    def print_stat(self):
        self.env.render()

    def reset(self):
        return self.env.reset()
