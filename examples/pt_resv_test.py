import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.components import PtReservoir
from noh.environments import Sin2CosTest
import matplotlib.pyplot as plt

if __name__ == "__main__":



    pt_resv = PtReservoir(1, 5, lr_type="hinton_r_div", r_div=100)
    env = Sin2CosTest(pt_resv)
    dataset = env.get_dataset()[0]

    pt_resv(dataset[0])

    env.train(epochs=1000)


    output = pt_resv(dataset)
    plt.plot(dataset)
    plt.plot(output)
    plt.show()


