from noh.components import Layer
from noh.components import RLTrainable
from noh.activate_functions import softmax
from noh.utils import get_discounted_rewards, get_standardized_rewards
import numpy as np


class PGLayerLuna(Layer, RLTrainable):
    def __init__(self, n_visible, n_hidden, is_return_id=True, is_argmax=False,
                 mbatch_size=10, lr=1e-9, decay_rate=0.99, gamma=0.9,
                 activate=softmax, reward_reset_checker=None):

        Layer.__init__(self, n_visible, n_hidden)
        RLTrainable.__init__(self, is_return_id=is_return_id, is_argmax=is_argmax,
                             mbatch_size=mbatch_size,
                             decay_rate=decay_rate, gamma=gamma,
                             reward_reset_checker=reward_reset_checker)

        self.lr = lr
        self.activate = activate

        print self.rl_trainable

    def __call__(self, data):

        act_prob = self.policy_prop(stat=data)
        if np.random.random() < self.epsilon:
            act_id = np.random.choice(range(self.n_hidden))
        else:
            if self.is_argmax:
                act_id = np.argmax(act_prob)
            else:
                act_id = np.random.choice(range(self.n_hidden), p=act_prob)
        act_vec = np.array([1 if i == act_id else 0 for i in xrange(self.n_hidden)])

        self.x_hist.append(data)
        self.d_logp_hist.append(act_vec - act_prob)
        self.reward_hist.append(None)
        self.reward_hist_landing.append(None)

        return act_id if self.is_return_id else act_prob

    def train(self):

        self.grad_counter += 1

        if self.grad_counter > self.mbatch_size:

            episode_x = np.vstack(self.x_hist)
            episode_logp = np.vstack(self.d_logp_hist)
            episode_reward = np.vstack(self.reward_hist)
            episode_reward_landing = np.vstack(self.reward_hist_landing)

            # episode_reward = self.get_standardized_rewards(episode_reward)
            episode_reward = get_discounted_rewards(episode_reward, self.gamma, self.reward_reset_checker)
            episode_reward_landing = get_discounted_rewards(episode_reward_landing, self.gamma,
                                                            self.reward_reset_checker) / 100.


            episode_logp *= (episode_reward + episode_reward_landing)
            self.dW += np.dot(episode_x.T, episode_logp)
            self.db_hidden += np.sum(episode_logp, axis=0)

            self.rmsprop_cache_W = self.decay_rate * self.rmsprop_cache_W + (1 - self.decay_rate) * self.dW**2
            self.W += self.lr * self.dW / (np.sqrt(self.rmsprop_cache_b_hidden) + 1e-5)
            self.dW = np.zeros_like(self.W)

            self.rmsprop_cache_b_hidden = self.decay_rate * self.rmsprop_cache_b_hidden + (1 - self.decay_rate) * self.db_hidden**2
            self.b_hidden += self.lr * self.db_hidden / (np.sqrt(self.rmsprop_cache_b_hidden) + 1e-5)
            self.db_hidden = np.zeros_like(self.b_hidden)

            self.grad_counter = 0

            self.x_hist, self.d_logp_hist, self.reward_hist, self.reward_hist_landing = [], [], [], []

    def set_reward(self, reward):
        if reward == 100 or reward == -100:
            self.reward_hist[-1] = 0
            self.reward_hist_landing[-1] = reward
        else:
            self.reward_hist[-1] = reward
            self.reward_hist_landing[-1] = 0

    def policy_prop(self, stat):
        h = self.activate(np.dot(stat, self.W) + self.b_hidden)
        return h
