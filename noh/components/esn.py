from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import Layer, Reservoir
from noh.training_functions import gen_linear_regression_trainer
from noh.activate_functions import linear

class ESNProp(PropRule):
    def __call__(self, data):
        components = self.components
        data = components.reservoir.prop_up_sequence(data)
        data = components.lin_reg(data)
        return data

class ESNTrain(TrainRule):
    def __call__(self, data, label, epochs):
        components = self.components
        data = components.reservoir.prop_up_sequence(data)
        return components.lin_reg.train(data, label, epochs=1)

class ESNPlanner(Planner):
    def __init__(self, components):
        super(ESNPlanner, self).__init__(
            components,
            prop=ESNProp,
            train=ESNTrain
        )

class ESN(Circuit):
    def __init__(self, n_visible, n_hidden, n_output, planner=ESNPlanner):
        super(ESN, self).__init__(
            reservoir=Reservoir(n_visible, n_hidden, bind_prob_W=0.03, bind_prob_W_rec=0.1),
            lin_reg=Layer(n_hidden, n_output, train_func_generator=gen_linear_regression_trainer, activate=linear),
            planner=planner
        )
