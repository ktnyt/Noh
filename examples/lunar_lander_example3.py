import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import LunarLander_v2
from noh.components import PGESN

n_stat = LunarLander_v2.n_stat
n_act = LunarLander_v2.n_act
reward_reset_checker = LunarLander_v2.gen_reward_reset_checker()
pgesn = PGESN(n_visible=n_stat, n_hidden=1000, n_output=n_act, is_argmax=False,
             reward_reset_checker=reward_reset_checker)
noh_env = LunarLander_v2(pgesn, render=True)

while True:
    noh_env.main_loop()
