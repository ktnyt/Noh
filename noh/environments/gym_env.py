import gym

from noh.environment import ReinforcementEnvironment


class GymEnv(ReinforcementEnvironment):

    n_stat = 0
    n_act = 0
    episode_size = 0

    @classmethod
    def gen_reward_reset_checker(cls):
        return lambda x: False

    def __init__(self, model, gym_env_name, render=False):
        super(GymEnv, self).__init__(model, render)
        self.env = gym.make(gym_env_name)
        self.action = None
        self.frame = 0
        self.default_positive_reward = 1.
        self.default_negative_reward = -1.

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
            observation, reward, done, info = self.step(action)
            self.model.set_reward(reward)
            episode_reward += reward
            if done: break

        if not done:
            self.model.set_reward(self.default_negative_reward)
            episode_reward += (-reward + self.default_negative_reward)

        self.model.train()
        self.episode_number += 1
        print ('ep %d: game finished, reward: %f' %
               (self.episode_number, episode_reward))

    def print_stat(self):
        self.env.render()

    def reset(self):
        return self.env.reset()


class Pong(GymEnv):
    n_stat = 80 * 80
    n_act = 6
    episode_size = 100000

    @classmethod
    def gen_reward_reset_checker(cls):
        return lambda x: True if x == 1 or x == -1 else False

    def __init__(self, model, render=False):
        super(Pong, self).__init__(model, "Pong-v0", render)

    def prepro(self, observation):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = observation
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.prepro(observation), reward, done, info

    def reset(self):
        return self.prepro(self.env.reset())


class LunarLander(GymEnv):

    n_stat = 8
    n_act = 4
    episode_size = 1000

    @classmethod
    def gen_reward_reset_checker(cls):
        return lambda x: True if x == 100 or x == -100 else False

    def __init__(self, model, render=False):
        super(LunarLander, self).__init__(model, "LunarLander-v2", render)
        self.default_positive_reward = 100
        self.default_negative_reward = -100