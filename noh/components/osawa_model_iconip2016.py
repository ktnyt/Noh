from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import PGLayer, RLTrainable
from noh.activate_functions import softmax


class StatProp(PropRule):
    def __call__(self, data):
        data = self.components.stl(data)
        data = self.components.output_layer(data)
        return data


class ActionTrain(TrainRule):
    def __call__(self):
        self.components.stl.train()
        self.components.output_layer.train()
        return None


class ICONIP2016Planner(Planner):
    def __init__(self, components):
        super(ICONIP2016Planner, self).__init__(
            components,
            PropRules={"stat_prop": StatProp},
            TrainRules={"action_train": ActionTrain},
            default_prop_name="stat_prop", default_train_name="action_train")

        self.is_stat_prop = False
        self.is_action_train = False

    def __call__(self, data):
        #print data, "->",
        act_vec = self.default_prop_rule(data)
        # act_id = np.argmax(act_vec)
        #print act_vec
        return act_vec

    def train(self, data=None, label=None, reward=None, epochs=None):
        self.default_train_rule()


class ModelICONIP2016(Circuit, RLTrainable):
    def __init__(self, stl, output_layer, planner=ICONIP2016Planner):
        Circuit.__init__(self, planner=planner, stl=stl, output_layer=output_layer)
        RLTrainable.__init__(self)

    def set_reward(self, reward):
        self.components.output_layer.set_reward(reward)

    def reset(self):
        self.components.stl.reset()