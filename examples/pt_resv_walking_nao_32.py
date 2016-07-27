import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.components import PtReservoir
from noh.environments import WalkingNao32Env
import matplotlib.pyplot as plt


if __name__ == "__main__":

    n_visible = WalkingNao32Env.n_visible

    pt_resv = PtReservoir(n_visible, 100, lr_type="hinton_r_div", r_div=1000)
    env = WalkingNao32Env(pt_resv)
    dataset = env.get_dataset()
    print dataset[0]
    pt_resv(dataset[0])

    env.train(epochs=1000)


    output = pt_resv(dataset)
    plt.plot(dataset)
    plt.plot(output)
    plt.show()


