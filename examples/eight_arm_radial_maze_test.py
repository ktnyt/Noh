import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.activate_functions import sigmoid
from noh.components import Reservoir, ModelICONIP2016
from noh.environments import EightArmRadialMaze

if __name__ == "__main__":

    n_visible = EightArmRadialMaze.n_stat
    n_hidden = 20
    n_output = EightArmRadialMaze.n_act
    n_reward = EightArmRadialMaze.n_reward
    n_reward_hidden = 10

    stl = Reservoir(n_visible=n_visible, n_hidden=n_hidden, activate=sigmoid, 
                    bind_prob_W=1., bind_prob_W_rec=0.3)
    model = ModelICONIP2016(n_visible=n_visible, n_hidden=n_hidden, n_output=n_output,
                            stl=stl)

    env = EightArmRadialMaze(model)
    while True:
        env.exec_episode()

    print env.stat_history
    print env.act_history
    print env.reward_history