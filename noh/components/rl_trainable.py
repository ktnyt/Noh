#from abc import ABCMeta, abstractmethod
import numpy as np

class RLTrainable(object):

    #__metaclass__ = ABCMeta

    def __init__(self, is_return_id=True, is_argmax=False, mbatch_size=10,
                 decay_rate=0.99, gamma=0.9, reward_reset_checker=None):

        self.rl_trainable = True

        self.is_return_id = is_return_id
        self.mbatch_size = mbatch_size
        self.decay_rate = decay_rate
        self.gamma = gamma

        if reward_reset_checker is None:
            reward_reset_checker = lambda x: False
        self.reward_reset_checker = reward_reset_checker

        self.x_hist, self.d_logp_hist, self.reward_hist, self.reward_hist_landing = [], [], [], []
        self.rmsprop_cache_W = np.zeros_like(self.W)
        self.rmsprop_cache_b_hidden = np.zeros_like(self.b_hidden)
        self.dW = np.zeros_like(self.W)
        self.db_hidden = np.zeros_like(self.b_hidden)
        self.grad_counter = 0
        self.epsilon = 0.9
        self.is_argmax = is_argmax
        self.prev_mean = 0.

    def set_reward(self, reward):
        self.reward_hist[-1] = reward