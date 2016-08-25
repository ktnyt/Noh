import numpy as np

from noh.components import Layer, RLTrainable
from noh.activate_functions import softmax
from noh.utils import get_standardized_rewards, get_discounted_rewards


class PGLayer(Layer, RLTrainable):

    def __init__(self, n_visible, n_hidden, is_return_id=True, is_argmax=False,
                 mbatch_size=10, epsilon=0.1, lr=1e-5, decay_rate=0.99, gamma=0.9,
                 activate=softmax, reward_reset_checker=None, is_output=False):

        Layer.__init__(self, n_visible, n_hidden)
        RLTrainable.__init__(self, is_return_id=is_return_id, is_argmax=is_argmax, mbatch_size=mbatch_size,
                             epsilon=epsilon, decay_rate=decay_rate, gamma=gamma,
                             reward_reset_checker=reward_reset_checker)

        self.lr = lr
        self.activate = activate
        self.is_output = is_output

    def __call__(self, data, epsilon=0.):

        self.epsilon = epsilon
        if self.is_output:

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

        else:
            output = self.policy_prop(stat=data)
            self.x_hist.append(data)
            return output

    def train(self, error=None):

        self.grad_counter += 1

        if error is None:
            if self.grad_counter >= self.mbatch_size:
                print "train!"
                episode_x = np.vstack(self.x_hist)
                episode_logp = np.vstack(self.d_logp_hist)
                episode_reward = np.vstack(self.reward_hist)

                #print episode_reward
                episode_reward = get_discounted_rewards(episode_reward, gamma=self.gamma,
                                                        reward_reset_checker=self.reward_reset_checker)
                #print episode_reward.flatten()
                episode_reward = get_standardized_rewards(episode_reward)

                episode_logp *= episode_reward
                self.grad["W"] += np.dot(episode_x.T, episode_logp)
                self.grad["b_hidden"] += np.sum(episode_logp, axis=0)

                self.rmsprop_cache["W"] = self.decay_rate * self.rmsprop_cache["W"] + (1 - self.decay_rate) * self.grad["W"]**2
                self.W += self.lr * self.grad["W"] / (np.sqrt(self.rmsprop_cache["W"]) + 1e-5)
                self.grad["W"] = np.zeros_like(self.W)

                #self.rmsprop_cache["b_hidden"] = self.decay_rate * self.rmsprop_cache["b_hidden"] + (1 - self.decay_rate) * self.grad["b_hidden"]**2
                #self.b_hidden += self.lr * self.grad["b_hidden"] / (np.sqrt(self.rmsprop_cache["b_hidden"]) + 1e-5)
                #self.grad["b_hidden"] = np.zeros_like(self.b_hidden)

                self.grad_counter = 0
                self.x_hist, self.d_logp_hist, self.reward_hist = [], [], []
                error_W = np.dot(episode_logp, self.W.T) * (1. * (episode_x > 0))
                error_b = np.dot(episode_logp, self.b_hidden)
                return (error_W, error_b)
        else:
            error_W, error_b = error
            episode_x = np.vstack(self.x_hist)
            self.grad["W"] += np.dot(episode_x.T, error_W)
            self.rmsprop_cache["W"] = self.decay_rate * self.rmsprop_cache["W"] + (1 - self.decay_rate) * self.grad["W"] ** 2
            self.W += self.lr * self.grad["W"] / (np.sqrt(self.rmsprop_cache["W"]) + 1e-5)
            self.grad["W"] = np.zeros_like(self.W)

            #self.grad["b_hidden"] += np.sum(error_b, axis=0)
            #self.rmsprop_cache["b_hidden"] = self.decay_rate * self.rmsprop_cache["b_hidden"] + (1 - self.decay_rate) * self.grad["b_hidden"] ** 2
            #self.b_hidden += self.lr * self.grad["b_hidden"] / (np.sqrt(self.rmsprop_cache["b_hidden"]) + 1e-5)
            #self.grad["b_hidden"] = np.zeros_like(self.b_hidden)

            self.grad_counter = 0
            self.x_hist, self.d_logp_hist, self.reward_hist = [], [], []
            new_error_W = np.dot(error_W, self.W.T) * (1. * (episode_x > 0))
            new_error_b = np.dot(error_W, self.b_hidden)
            return (new_error_W, new_error_b)

    def set_reward(self, reward):
        self.reward_hist[-1] = reward

    def policy_prop(self, stat):
        h = self.activate(np.dot(stat, self.W) + self.b_hidden)
        return h



