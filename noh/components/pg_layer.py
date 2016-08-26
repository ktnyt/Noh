import numpy as np

from noh.components import Layer, RLTrainable
from noh.activate_functions import softmax
from noh.utils import get_standardized_rewards, get_discounted_rewards


class PGLayer(Layer, RLTrainable):

    def __init__(self, n_visible, n_hidden, is_return_id=False, is_argmax=False,
                 mbatch_size=10, lr=1e-5, decay_rate=0.99, gamma=0.9,
                 activate=softmax, reward_reset_checker=None, is_output=False):

        Layer.__init__(self, n_visible, n_hidden)
        RLTrainable.__init__(self, is_return_id=is_return_id, is_argmax=is_argmax, mbatch_size=mbatch_size,
                             decay_rate=decay_rate, gamma=gamma,
                             reward_reset_checker=reward_reset_checker)

        self.lr = lr
        self.activate = activate
        self.is_output=is_output

    def __call__(self, data, epsilon=0.):

        self.x_hist.append(data)
        act_prob = self.policy_prop(stat=data)
        if not self.is_output:
            return act_prob

        if np.random.random() < epsilon:
            act_id = np.random.choice(range(self.n_hidden))
        elif self.is_argmax:
            act_id = np.argmax(act_prob)
        else:
            act_id = np.random.choice(range(self.n_hidden), p=act_prob)
        act_vec = np.array([1 if i == act_id else 0 for i in xrange(self.n_hidden)])

        self.d_logp_hist.append(act_vec - act_prob)
        self.reward_hist.append(None)
        return act_id if self.is_return_id else act_prob

    def train(self, error=None):

        self.grad_counter += 1

        episode_x = np.vstack(self.x_hist)
        if error is None:
            if self.grad_counter < self.mbatch_size:
                return

            error = np.vstack(self.d_logp_hist)
            episode_reward = np.vstack(self.reward_hist)
            episode_reward = get_discounted_rewards(episode_reward, gamma=self.gamma,
                                                    reward_reset_checker=self.reward_reset_checker)
            episode_reward = get_standardized_rewards(episode_reward)
            error *= episode_reward

        self.grad["W"] += np.dot(episode_x.T, error)
        self.grad["b_hidden"] += np.sum(error, axis=0)
        for key in ["W"]:
            self.rmsprop_cache[key] = self.decay_rate * self.rmsprop_cache[key] + (1 - self.decay_rate) * self.grad[key]**2
            param = self.params[key]()
            param += self.lr * self.grad[key] / (np.sqrt(self.rmsprop_cache[key]) + 1e-5)
            #self.grad[key] = np.zeros_like(param)
            self.grad[key] *= 0.95

        self.grad_counter = 0
        self.x_hist, self.d_logp_hist, self.reward_hist = [], [], []
        error = np.dot(error, self.W.T) * (1. * (episode_x > 0))
        # error_b = np.dot(error, self.b_hidden)
        return error

    def set_reward(self, reward):
        self.reward_hist[-1] = reward

    def policy_prop(self, stat):
        h = self.activate(np.dot(stat, self.W) + self.b_hidden)
        return h



