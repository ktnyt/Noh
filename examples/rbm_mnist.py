import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np

from noh.components import RBM
from noh.environments import MNISTEnv

if __name__ == "__main__":

    # rbm = RBM(6, 2, lr_type="const", lr=0.2)
    rbm = RBM(28*28, 50, lr_type="hinton_r_div", r_div=100)
    env = MNISTEnv(rbm)

    env.show_test_images()

    env.train(epochs=5)

    env.show_reconstruct_images()

    for i in xrange(10):
        dataset = rbm.gen_sampled_data(n_sample=25, sampling_epochs=i)
        env.show_images(dataset)

    