import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.components import RBM
from noh.environments import SimpleUnsupervisedTest

if __name__ == "__main__":

    rbm = RBM(6, 2)
    env = SimpleUnsupervisedTest(rbm)
    env.train()
    print rbm.rec(env.get_dataset())
    print rbm.rec(env.get_test_dataset())

    
