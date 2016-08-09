import gym
import numpy as np

from noh.environment import ReinforcementEnvironment
from noh.activate_functions import softmax


class LunarLander_v2(ReinforcementEnvironment):
    n_stat = 8
    n_act = 4
    @classmethod
    def gen_reward_reset_checker(cls):
        return lambda x: True if x == 100 or x == -100 else False

    def __init__(self, model, render=False):
        self.model = model
        self.render = render
        self.env = gym.make("LunarLander-v2")
        self.observation = self.env.reset()
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.batch_size = 10
        self.action = None
        self.frame = 0

    def get_reward(self):
        return self.reward

    def get_stat(self):
        return self.observation

    def set_act(self, action):
        return self.env.step(action)

    def main_loop(self):
        if self.render: self.env.render()

        stat = self.observation
        if self.frame % 1 == 0:
            self.action = self.model(stat)
        self.observation, reward, done, info = self.env.step(self.action)
        self.model.set_reward(reward)
        self.reward_sum += reward

        # an episode finished
        if done:
            self.model.train()
            self.episode_number += 1
            print ('ep %d: game finished, reward: %f' % (self.episode_number, self.reward_sum))
            self.observation = self.env.reset()
            self.reward_sum = 0
            self.frame = 0
