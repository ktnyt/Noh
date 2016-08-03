import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.environments import LunarLander_v2
from noh.components import RLNN

n_stat = LunarLander_v2.n_stat
n_act = LunarLander_v2.n_act

rlnn = RLNN(n_visible=n_stat, n_hidden=200, n_output=n_act, resume=True)
noh_env = LunarLander_v2(rlnn, render=True)

while True:
    noh_env.main_loop()
    # preprocess the observation, set input to network to be difference image