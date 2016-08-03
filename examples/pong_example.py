import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.environments import Pong
from noh.components import RLNN


n_stat = Pong.n_stat
n_act = Pong.n_act
rlnn = RLNN(n_visible=n_stat, n_hidden=200, n_output=n_act, resume=True)
noh_env = Pong(rlnn, render=True)

while True:
    noh_env.main_loop()
    # preprocess the observation, set input to network to be difference image
