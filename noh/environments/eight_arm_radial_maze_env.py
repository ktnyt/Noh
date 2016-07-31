from noh.environment import ReinforcementEnvironment

import numpy as np


class EightArmRadialMaze(ReinforcementEnvironment):

    n_act = 9
    n_stat = 9
    n_reward = 1

    map = []
    for line in open("../noh/environments/eight_arm_radial_maze_map.dat"):
        map.append([int(d) for d in line[:-1].split(" ")])

    map_len = (len(map), len(map[0]))

    default_pos = (7, 7)
    default_reward_pos_list = [(1, 1), (1, 7), (1, 13),
                               (7, 1), (7, 7), (7, 13),
                               (13, 1), (13, 7), (13, 13)]
    act_list = [(-6, -6), (-6, 0), (-6, 6),
                (0, -6), (0, 0), (0, 6),
                (6, -6), (6, 0), (6, 6)]

    episode_size = 15

    def __init__(self, model):
        super(EightArmRadialMaze, self).__init__(model)
        self.pos = self.reward_pos_list = None
        self.reward = 0
        self.reset()

        self.act_history = []
        self.stat_history = []
        self.reward_history = []

    @classmethod
    def print_map(cls):
        for line in cls.map:
            for area in line:
                print area,
            print " "

    def train(self, epochs):
        for epoch in xrange(epochs):
            self.reset()
            self.main_loop()

    def main_loop(self):
        for _ in xrange(self.episode_size):
            stat = self.get_stat()
            self.stat_history.append(stat)

            act = self.model(stat)
            self.act_history.append(act)
            self.set_act(act)

            reward = self.get_reward()
            self.reward_history.append(reward)

    def reset(self):
        self.pos = self.default_pos
        self.reward_pos_list = self.default_reward_pos_list[:]
        self.reward = 0

    def get_stat(self):
        """ Return a vector """
        return self.pos

    def set_act(self, act_id):
        """ set a vector """
        act = self.act_list[act_id]
        print self.pos, act, "->",

        punctuation = 2
        new_y = self.pos[0]
        new_x = self.pos[1]
        for i in xrange(punctuation):

            new_y += (act[0] / punctuation)
            new_x += (act[1] / punctuation)

            if (new_y < 0 or new_x < 0 or
                    new_y > self.map_len[0] or new_x > self.map_len[1] or
                    self.map[new_y][new_x] == 0):
                self.reward = -1
                print "fail"
                return

        self.pos = (new_y, new_x)
        self.reward = 0

        print self.pos, self.reward

    def get_reward(self):
        """ Return some scholar value """
        t_pos = tuple(self.pos)
        if t_pos in self.reward_pos_list:
            self.reward_pos_list.remove(t_pos)
            return 1
        return 0

    def print_stat(self):
        for y, line in enumerate(self.map):
            for x, area in enumerate(line):
                if y == self.pos[0] and x == self.pos[1]:
                    print "M",
                elif (y, x) in self.reward_pos_list:
                    print "R",
                elif area == 0:
                    print " ",
                elif area == 1:
                    print "*",
            print " "