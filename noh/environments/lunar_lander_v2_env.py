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
        self.prev_x = None  # used in computing the difference frame
        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.batch_size = 10  # every how many episodes to do a param update?


    def get_reward(self):
        pass


    def get_stat(self):
        pass


    def set_act(self):
        pass


    def main_loop(self):
        if self.render: self.env.render()

        cur_x = self.prepro()
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.__class__.n_stat)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = self.model(x)
        action = np.random.choice(range(self.__class__.n_act), p=softmax(aprob))
        # action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        # record various intermediates (needed later for backprop)
        self.xs.append(x)  # observation
        self.hs.append(h)  # hidden state
        # y = 1 if action == 2 else 0  # a "fake label"
        y = np.array([1 if action == i else 0 for i in xrange(self.n_act)])
        self.dlogps.append(
            y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurements
        self.observation, reward, done, info = self.env.step(action)
        self.reward_sum += reward

        self.drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:  # an episode finished
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
            print ('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + (
            '' if reward == -1 else ' !!!!!!!!')


    def prepro(self):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = self.observation
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()


    def discount_rewards(self, r, gamma=0.9):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r