import gym
import numpy as np

from noh.environment import ReinforcementEnvironment
from noh.activate_functions import softmax


class LunarLander_v2(ReinforcementEnvironment):

    n_stat = 8
    n_act = 4

    def __init__(self, model, render=False):
        self.model = model
        self.render = render
        self.env = gym.make("LunarLander-v2")
        self.observation = self.env.reset()
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.batch_size = 10

    def get_reward(self):
        return self.reward


    def get_stat(self):
        return self.observation


    def set_act(self, action):
        return self.env.step(action)


    def main_loop(self):
        if self.render: self.env.render()

        x = self.get_stat()
        
        aprob, h = self.model(x)
        action = np.random.choice(range(self.__class__.n_act), p=softmax(aprob))
        print aprob
        self.xs.append(x)
        self.hs.append(h)

        y = np.array([1 if action == i else 0 for i in xrange(self.n_act)])
        self.dlogps.append(y - aprob)
        # step the environment and get new measurements

        self.observation, reward, done, info = self.set_act(action)

        # reward = self.get_reward()
        self.reward_sum += reward

        self.drs.append(reward)  

        # an episode finished
        if done:  
            self.episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(self.xs)
            eph = np.vstack(self.hs)
            epdlogp = np.vstack(self.dlogps)
            epr = np.vstack(self.drs)
            self.xs, self.hs, self.dlogps, self.drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = self.discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
            self.model.policy_backward(epx, eph, epdlogp)

            # perform rmsprop parameter update every batch_size episodes
            if self.episode_number % self.batch_size == 0:
                self.model.train()
            # boring book-keeping
            self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward)
            if self.episode_number % 20 == 0: self.model.save()
            self.reward_sum = 0
            self.observation = self.env.reset()  # reset env
            self.prev_x = None

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (self.episode_number, reward))

    def discount_rewards(self, r, gamma=0.9):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            if r[t] == 100 or r[t] == -100: running_add = 0
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
