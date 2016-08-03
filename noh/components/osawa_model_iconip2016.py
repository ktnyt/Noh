from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import Layer, Reservoir, RBM
from noh.training_functions import gen_sgd_trainer
from noh.activate_functions import softmax

import numpy as np

class StatProp(PropRule):
    def __call__(self, data):
        print data
        data = self.components.stl(data)
        data = self.components.softmax_layer(data)
        print data
        return data


class RewardProp(PropRule):
    def __call__(self, data):
        data = self.components.stl(data)
        energy = self.components.reward_rbm.get_energy(data)
        return energy


class ActionTrain(TrainRule):
    def __call__(self, data, label, reward, epochs):
        data = self.components.stl(data)
        stl_error = self.components.stl.unsupervised_train(data, epochs=10)
        data = self.components.softmax_layer(data)
        softmax_error = self.components.softmax_layer.supervised_train(data, epochs=10)
        return stl_error, softmax_error


class RewardTrain(TrainRule):
    def __call__(self, data, label, reward, epochs):
        data = self.components.stl(data)
        stl_error = self.components.stl.unsupervised_train(data, epochs=10)
        data = self.components.softmax_layer(data)
        softmax_error = self.components.softmax_layer.supervised_train(data, epochs=10)
        return stl_error, softmax_error


class ICONIP2016Planner(Planner):
    def __init__(self, components):
        super(ICONIP2016Planner, self).__init__(
            components,
            PropRules={"stat_prop": StatProp, "reward_prop": RewardProp},
            TrainRules={"action_train": ActionTrain, "reward_train": RewardTrain},
            default_prop_name="stat_prop", default_train_name="action_train")

        self.is_stat_prop = False
        self.is_action_train = False

    def __call__(self, data):
        act_vec = self.default_prop_rule(data)
        # act_id = np.argmax(act_vec)
        return act_vec

    def train(self, data, label, reward, epochs):
        pass

class ModelICONIP2016(Circuit):
    def __init__(self, n_visible, n_hidden, n_output, n_reward_hidden, stl, planner=ICONIP2016Planner):
        super(ModelICONIP2016, self).__init__(
            planner=planner,             
            RL_trainable=True,
            stl=stl,
            softmax_layer=Layer(n_hidden, n_output, train_func_generator=gen_sgd_trainer, activate=softmax),
            reward_rbm=RBM(n_visible=n_hidden, n_hidden=n_reward_hidden, lr_type="hinton_r_div", r_div=100)
        )
