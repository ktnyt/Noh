import numpy as np

from noh import Circuit, Planner
from noh.components import Layer, Reservoir
from noh.training_functions import gen_linear_regression_trainer
from noh.activate_functions import liner

class ESN_Planner(Planner):
    def __init__(self, components):
        super(ESN_Planner, self).__init__(components)

    def __call__(self, data):
        data = self.components[0].prop_up_sequence(data)
        data = self.components[1](data)
        return data

    def train(self, data, label, epochs):
        resv = self.components[0].prop_up_sequence(data)
        self.components[1].train(resv, label, epochs=1)
        

class ESN(Circuit):
    def __init__(self, n_visible, n_hidden, n_output):
        super(ESN, self).__init__(components=None, planner=None)

        resv = Reservoir(n_visible, n_hidden, bind_prob_W=0.03, bind_prob_W_rec=0.1)
        liner_regression = Layer(n_hidden, n_output, train_func_generator=gen_linear_regression_trainer, 
                                 activate=liner)
        
        self.components = (resv, liner_regression)
        self.planner = ESN_Planner(self.components)
