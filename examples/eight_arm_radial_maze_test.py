import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.components import RLTester
from noh.environments import EightArmRadialMaze

if __name__ == "__main__":

    rl_tester = RLTester()
    env = EightArmRadialMaze(rl_tester)
    env.train(epochs=2)
