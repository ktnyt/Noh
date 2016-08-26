import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.activate_functions import sigmoid, softmax
from noh.components import Reservoir, PGLayer, ModelICONIP2016
from noh.environments import EightArmRadialMaze

if __name__ == "__main__":

    n_visible = EightArmRadialMaze.n_stat
    n_hidden = 20
    n_output = EightArmRadialMaze.n_act
    n_reward = EightArmRadialMaze.n_reward

    reward_reset_checker = EightArmRadialMaze.gen_reward_reset_checker()

    stl = Reservoir(n_visible=n_visible, n_hidden=n_hidden, activate=sigmoid, 
                    bind_prob_W=0.3, bind_prob_W_rec=0.3)
    output_layer = PGLayer(n_visible=n_hidden, n_hidden=n_output, is_return_id=True, is_argmax=False,
                           mbatch_size=10, epsilon=0, lr=1e-7, decay_rate=0.9, gamma=0.5, activate=softmax,
                           reward_reset_checker=reward_reset_checker)
    model = ModelICONIP2016(stl=stl, output_layer=output_layer)

    env = EightArmRadialMaze(model)
    f_rewards = open("eight_arm_radial_maze_rewards.dat", "w")
    while True\
            :
        env.exec_episode()
        f_rewards.write(str(env.reward_history[-1]) + "\n")

    f_rewards.close()
