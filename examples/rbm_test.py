import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.components import RBM
from noh.environments import SimpleUnsupervisedTest

if __name__ == "__main__":

    # rbm = RBM(6, 2, lr_type="const", lr=0.2)
    rbm = RBM(6, 2, lr_type="hinton_r_div", r_div=10)
    env = SimpleUnsupervisedTest(rbm)
    env.train(epochs=1000)
    print rbm.rec(env.get_dataset())
    print rbm.rec(env.get_test_dataset())

    
