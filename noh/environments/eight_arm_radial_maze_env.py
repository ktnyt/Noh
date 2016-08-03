from noh.environments import BehavioralTestBattery

import numpy as np


class EightArmRadialMaze(BehavioralTestBattery):

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

    def __init__(self, model, act_punctuation=2):
        super(EightArmRadialMaze, self).__init__(model, act_punctuation)

    def pos2id(self, pos):
        res = np.zeros(shape=(3, 3))
        res[(pos[0]-1)/6][(pos[1]-1)/6] = 1.
        return res.flatten()