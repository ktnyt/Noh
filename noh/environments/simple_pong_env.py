from noh.environment import ReinforcementEnvironment
import gym
import numpy as np
from noh.activate_functions import softmax

class Pong(ReinforcementEnvironment):
    # model initialization
    n_stat = 80 * 80  # input dimensionality: 80x80 grid
    n_act = 6
    episode_size = 100000

    @classmethod
    def gen_reward_reset_checker(cls):
        return lambda x: True if x == 1 or x == -1 else False

    def __init__(self, model, render=False):
        super(Pong, self).__init__(model, render)
        self.env = gym.make("Pong-v0")
        self.batch_size = 10
        self.action = None
        self.frame = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.prepro(observation), reward, done, info

    def exec_episode(self):
        episode_reward = 0
        observation = self.reset()
        for frame in xrange(self.__class__.episode_size):
            if self.render: self.print_stat()
            action = self.model(observation)
            observation, reward, done, info = self.step(action)
            self.model.set_reward(reward)
            episode_reward += reward
            if done: break

        self.model.train()
        self.episode_number += 1

        print ('ep %d: game finished, reward: %f' %
               (self.episode_number, episode_reward))

    def prepro(self, observation):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = observation
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def print_stat(self):
        self.env.render()

    def reset(self):
        return self.prepro(self.env.reset())