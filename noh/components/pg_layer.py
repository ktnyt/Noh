import numpy as np

from noh.components import Layer
from noh.activate_functions import softmax


class PGLayer(Layer):
    def __init__(self, n_visible, n_hidden, is_return_id=True, is_argmax=False,
                 mbatch_size=10, lr=1e-5, decay_rate=0.99, gamma=0.9,
                 activate=softmax, reward_reset_checker=None):

        super(PGLayer, self).__init__(n_visible, n_hidden)
        self.is_return_id = is_return_id
        self.mbatch_size = mbatch_size
        self.lr = lr
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.activate = activate
        if reward_reset_checker is None:
            reward_reset_checker = lambda x: False
        self.reward_reset_checker = reward_reset_checker

        self.x_hist, self.d_logp_hist, self.reward_hist = [], [], []
        self.rmsprop_cache_W = np.zeros_like(self.W)
        self.rmsprop_cache_b_hidden = np.zeros_like(self.b_hidden)
        self.dW = np.zeros_like(self.W)
        self.db_hidden = np.zeros_like(self.b_hidden)
        self.grad_counter = 0
        self.epsilon = 0.1
        self.is_argmax = is_argmax
        self.prev_mean = 0.

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

        return act_id if self.is_return_id else act_prob

    def train(self):

        self.grad_counter += 1

        if self.grad_counter > self.mbatch_size:

            episode_x = np.vstack(self.x_hist)
            episode_logp = np.vstack(self.d_logp_hist)
            episode_reward = np.vstack(self.reward_hist)

            episode_reward = self.get_standardized_rewards(episode_reward)
            # episode_reward = self.get_discounted_rewards(episode_reward)


            episode_logp *= episode_reward
            self.dW += np.dot(episode_x.T, episode_logp)
            self.db_hidden += np.sum(episode_logp, axis=0)

            self.rmsprop_cache_W = self.decay_rate * self.rmsprop_cache_W + (1 - self.decay_rate) * self.dW**2
            self.W += self.lr * self.dW / (np.sqrt(self.rmsprop_cache_b_hidden) + 1e-5)
            self.dW = np.zeros_like(self.W)

            self.rmsprop_cache_b_hidden = self.decay_rate * self.rmsprop_cache_b_hidden + (1 - self.decay_rate) * self.db_hidden**2
            self.b_hidden += self.lr * self.db_hidden / (np.sqrt(self.rmsprop_cache_b_hidden) + 1e-5)
            self.db_hidden = np.zeros_like(self.b_hidden)

            self.grad_counter = 0


            self.x_hist, self.d_logp_hist, self.reward_hist = [], [], []

    def set_reward(self, reward):
        self.reward_hist[-1] = reward

    def policy_prop(self, stat):
        h = self.activate(np.dot(stat, self.W) + self.b_hidden)
        return h

    def get_standardized_rewards(self, rewards):
        #res_rewards = self.get_discounted_rewards(rewards)
        res_rewards = rewards

        res_rewards -= np.mean(res_rewards)
        res_rewards /= np.std(res_rewards)
        return res_rewards

    def get_discounted_rewards(self, rewards):
        discounted_reward_list = np.zeros_like(rewards)
        discounted_reward = 0.
        for t in reversed(xrange(0, rewards.size)):
            if self.reward_reset_checker(self.reward_hist[t]):
                discounted_reward = 0.
            discounted_reward = (discounted_reward * self.gamma) + rewards[t]
            discounted_reward_list[t] = discounted_reward

        return discounted_reward_list

