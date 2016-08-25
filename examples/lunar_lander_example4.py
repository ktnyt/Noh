import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import LunarLander_v2
from noh.components import PGNet

n_stat = LunarLander_v2.n_stat
n_act = LunarLander_v2.n_act
reward_reset_checker = LunarLander_v2.gen_reward_reset_checker()
pg_net = PGNet(structure=[n_stat*4, 500, 500, 500, n_act],
               reward_reset_checker=reward_reset_checker, is_argmax=True)
noh_env = LunarLander_v2(pg_net, render=True)

while True:
    noh_env.exec_episode()