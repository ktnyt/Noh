from abc import ABCMeta, abstractmethod
from noh.environment import ReinforcementEnvironment

import numpy as np


class BehavioralTestBattery(ReinforcementEnvironment):

    n_act = -1
    n_stat = -1
    n_reward = -1

    map = []
    map_len = (0, 0)

    default_pos = (0, 0)
    default_reward_pos_list = []
    act_list = []

    episode_size = 0

    def __init__(self, model, act_punctuation=3):
        super(BehavioralTestBattery, self).__init__(model)
        self.pos = self.reward_pos_list = None
        self.reward = 0
        self.reset()

        self.act_history = []
        self.act_prob_history = []
        self.stat_history = []
        self.reward_history = []
        self.act_punctuation = act_punctuation

        self.sum_reward = 0

    @classmethod
    def print_map(cls):
        for line in cls.map:
            for area in line:
                print area,
            print " "

    def step(self, act_id):
        """ set a vector """
        act = self.__class__.act_list[act_id]
        new_y = self.pos[0]
        new_x = self.pos[1]
        reward = 0

        for i in xrange(self.act_punctuation):

            new_y += (act[0] / self.act_punctuation)
            new_x += (act[1] / self.act_punctuation)
            new_pos = (new_y, new_x)
            if (new_y < 0 or new_x < 0 or
                new_y > self.__class__.map_len[0] or new_x > self.__class__.map_len[1] or
                self.__class__.map[new_y][new_x] == 0):
                """" out of map """
                return self.pos2id(self.pos), -1, False, None
            elif new_pos in self.reward_pos_list:
                """ get reward """
                self.reward_pos_list.remove(new_pos)
                reward += 1.
            elif self.__class__.map[new_y][new_x] == 2:
                """ unrewarding place"""
                reward = -1.
            else:
                """ ordinal place"""
                reward = 0.
        self.pos = new_pos
        return self.pos2id(self.pos), reward, False, None

    def exec_episode(self):
        episode_reward = 0
        episode_acts = []
        episode_stats = []
        observation = self.reset()

        for frame in xrange(self.__class__.episode_size):
            if self.render: self.print_stat()
            action = self.model(observation)
            observation, reward, done, info = self.step(action)
            self.model.set_reward(reward)

            episode_reward += reward
            episode_acts.append(action)
            episode_stats.append(observation)

            if done: break

        self.model.train()

        self.reward_history.append(episode_reward)
        self.act_history.append(episode_acts)
        self.stat_history.append(episode_stats)

        print ("ep %d: game finished, reward: %f" %
               (self.episode_number, episode_reward))
        self.episode_number += 1

    def pos2id(self, pos):
        res = np.zeros(shape=self.__class__.map_len)
        res[pos[0]][pos[1]] = 1.
        return res.flatten()

    def reset(self):
        self.pos = self.__class__.default_pos
        self.reward_pos_list = self.__class__.default_reward_pos_list[:]
        self.reward = 0
        self.model.reset()
        return self.pos2id(self.pos)

    def print_stat(self):
        print "- - - - - - - - - - - - - - - - -"
        for y, line in enumerate(self.__class__.map):
            print "-",
            for x, area in enumerate(line):
                if y == self.pos[0] and x == self.pos[1]:
                    print "M",
                elif (y, x) in self.reward_pos_list:
                    print "r",
                elif area == 0:
                    print "-",
                elif area == 1:
                    print " ",
                elif area == 2:
                    print "~",
            print "-"
        print "- - - - - - - - - - - - - - - - -"
