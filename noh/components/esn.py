from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import Layer, Reservoir
from noh.training_functions import gen_linear_regression_trainer
from noh.activate_functions import linear

class ESNProp(PropRule):
    def __call__(self, data):
        data = self.reservoir.prop_up_sequence(data)
        data = self.layer(data)
        return data

class ESNTrain(TrainRule):
    def __call__(self, data, label, epochs):
        data = self.reservoir.prop_up_sequence(data)
        data = self.layer.train(data, label, epochs=1)
        return data

class ESNPlanner(Planner):
    def __init__(self, components):
        super(ESNPlanner, self).__init__(
            prop=ESNProp,
            train=ESNTrain,
            components=components
        )

class ESN(Circuit):
    def __init__(self, components, planner=ESNPlanner):
        super(ESN, self).__init__(
            planner=planner,
            components=components
        )

    @classmethod
    def create(cls, n_visible, n_hidden, n_output, planner=ESNPlanner):
        return cls(components=[
            Reservoir(n_visible, n_hidden, bind_prob_W=0.03, bind_prob_W_rec=0.1),
            Layer(n_hidden, n_output, train_func_generator=gen_linear_regression_trainer, activate=linear)
        ], planner=planner)
