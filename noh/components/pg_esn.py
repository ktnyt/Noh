from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import PGLayerLuna, Reservoir
from noh.training_functions import gen_linear_regression_trainer
from noh.activate_functions import linear


class PGESNProp(PropRule):
    def __call__(self, data):
        components = self.components
        #data = components.reservoir.prop_up_sequence(data)
        data = components.reservoir(data)
        data = components.pg_layer(data)
        return data


class PGESNTrain(TrainRule):
    def __call__(self):
        self.components.pg_layer.train()


class PGESNPlanner(Planner):
    def __init__(self, components):
        super(PGESNPlanner, self).__init__(
            components,
            PropRules={"default":PGESNProp},
            TrainRules={"default":PGESNTrain})

    def train(self):
        return self.default_train_rule()


class PGESN(Circuit):

    def __init__(self, n_visible, n_hidden, n_output, planner=PGESNPlanner,
                 is_argmax=False,
                 reward_reset_checker=None):
        super(PGESN, self).__init__(
            reservoir=Reservoir(n_visible, n_hidden, 
                                bind_prob_W=0.3, bind_prob_W_rec=0.2),
            pg_layer=PGLayerLuna(n_visible=n_hidden, n_hidden=n_output,
                             is_return_id=True, is_argmax=is_argmax, mbatch_size=10,
                             lr=1e-9, decay_rate=0.99, gamma=0.9,
                             reward_reset_checker=reward_reset_checker),
            planner=planner)

    def train(self):
        return self.planner.train()

    def set_reward(self, reward):
        self.components.pg_layer.set_reward(reward)
