import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.components import ESN
from noh.environments import Sin2CosTest
import matplotlib.pyplot as plt

if __name__ == "__main__":
    esn = ESN(1, 100, 1)
    env = Sin2CosTest(esn)
    env.train()
    output = esn(env.get_dataset()[0])

    plt.plot(Sin2CosTest.get_dataset()[0])
    plt.plot(output)
    plt.show()


