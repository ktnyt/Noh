import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh.components import Layer
from noh.environments import SimpleTest

if __name__ == "__main__":
    layer = Layer(10, 10)
    env = SimpleTest(layer)
    env.train()
    print layer(env.get_dataset()[0])

    
