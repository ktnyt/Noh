import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import LunarLander_v2
from noh.components import PGLayerLuna

n_stat = LunarLander_v2.n_stat
n_act = LunarLander_v2.n_act
reward_reset_checker = LunarLander_v2.gen_reward_reset_checker()
rlnn = PGLayerLuna(n_visible=n_stat, n_hidden=n_act, reward_reset_checker=reward_reset_checker, is_argmax=False)
noh_env = LunarLander_v2(rlnn, render=True)

while True:
    noh_env.main_loop()
