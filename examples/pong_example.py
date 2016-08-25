import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.environments import Pong
from noh.components import PGNet


n_stat = Pong.n_stat
n_act = Pong.n_act
reward_reset_checker = Pong.gen_reward_reset_checker()
pg_net = PGNet(structure=[n_stat, 100, 100, 100, n_act], reward_reset_checker=reward_reset_checker, is_argmax=False)
noh_env = Pong(pg_net, render=False)

while True:
    noh_env.exec_episode()