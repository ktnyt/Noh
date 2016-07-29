from noh import Circuit, Router
from noh.routes import PropRoute, TrainRoute
from noh.components import Layer, Reservoir
from noh.training_functions import gen_linear_regression_trainer
from noh.activate_functions import linear

class ESNProp(PropRoute):
    def __call__(self, data):
        components = self.components
        data = components.reservoir.prop_up_sequence(data)
        data = components.lin_reg(data)
        return data

class ESNTrain(TrainRoute):
    def __call__(self, data, label, epochs):
        components = self.components
        data = components.reservoir.prop_up_sequence(data)
        return components.lin_reg.train(data, label, epochs=1)

class ESN(Circuit):
    def __init__(self, n_visible, n_hidden, n_output):
        super(ESN, self).__init__(
            reservoir=Reservoir(n_visible, n_hidden, bind_prob_W=0.03, bind_prob_W_rec=0.1),
            lin_reg=Layer(n_hidden, n_output, train_func_generator=gen_linear_regression_trainer, activate=linear),
            router=Router(
                prop=ESNProp(),
                train=ESNTrain(),
            )
        )
