from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import PGLayer
from noh.activate_functions import softmax


class StatProp(PropRule):
    def __call__(self, data):
        print data
        data = self.components.stl(data)
        data = self.components.softmax_layer(data)
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

    def train(self, data=None, label=None, reward=None, epochs=None):
        pass


class ModelICONIP2016(Circuit):
    def __init__(self, n_visible, n_hidden, n_output, stl, planner=ICONIP2016Planner):
        super(ModelICONIP2016, self).__init__(
            planner=planner,             
            RL_trainable=True,
            stl=stl,
            softmax_layer=PGLayer(n_visible=n_hidden, n_hidden=n_output,
                                  is_return_id=True, is_argmax=False,
                                  mbatch_size=200, lr=0.01, decay_rate=0.9, gamma=0.9,
                                  activate=softmax, reward_reset_checker=None),
            rl_trainable=True)
