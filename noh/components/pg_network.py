from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import PGLayer, RLTrainable
from noh.activate_functions import relu, p_relu, softmax
import numpy as np

class PGNetProp(PropRule):

    def __init__(self, components):
        super(PGNetProp, self).__init__(components)
        self.size = 4
        self.prev_input = None
        self.prev_output = None
        self.counter = 0

    def __call__(self, input_data, epsilon):

        if self.counter % 1 != 0:
            return self.prev_output

        if self.prev_input is None:
            self.prev_input = [np.zeros_like(input_data) for n in xrange(self.size-1)]

        data = input_data
        for prev_d in self.prev_input:
            data = np.r_[data, prev_d]

        for component in [self.components.layer0, self.components.layer1, self.components.layer2]:
            data = component(data)

        data = self.components.layer3(data, epsilon)


        del self.prev_input[0]
        self.prev_input.append(input_data)
        self.prev_output = data
        return data

class PGNetTrain(TrainRule):
    def __call__(self):
        error = None
        for component in [self.components.layer3, self.components.layer2,
                          self.components.layer1, self.components.layer0]:
            error = component.train(error=error)



class PGNetPlanner(Planner):

    def __init__(self, components):
        super(PGNetPlanner, self).__init__(
            components,
            PropRules={"default":PGNetProp},
            TrainRules={"default":PGNetTrain})

    def __call__(self, input_data, epsilon):
        return self.default_prop_rule(input_data, epsilon)

    def train(self, data=None, label=None, epochs=None):
        return self.default_train_rule()

class PGNet(Circuit, RLTrainable):

    def __init__(self, structure, planner=PGNetPlanner,
                 mbatch_size=10, lr=1e-5, epsilon=0., decay_rate=0.99, gamma=0.99,
                 is_argmax=False, reward_reset_checker=None):
        Circuit.__init__(self, planner=planner,
            layer0=PGLayer(n_visible=structure[0], n_hidden=structure[1],
                           lr=lr, decay_rate=decay_rate, activate=relu),
            layer1=PGLayer(n_visible=structure[1], n_hidden=structure[2],
                           lr=lr, decay_rate=decay_rate, activate=relu),
            layer2=PGLayer(n_visible=structure[2], n_hidden=structure[3],
                           lr=lr, decay_rate=decay_rate, activate=relu),
            layer3=PGLayer(n_visible=structure[3], n_hidden=structure[4],
                           is_return_id=True, is_argmax=is_argmax,
                           mbatch_size=mbatch_size, lr=lr, decay_rate=decay_rate, gamma=gamma,
                           activate=softmax, reward_reset_checker=reward_reset_checker, is_output=True))
        RLTrainable.__init__(self)

        self.epsilon = epsilon

    def __call__(self, data):
        self.epsilon -= 1./1e5
        if self.epsilon < 0.01:
            self.epsilon = 0.01
        return self.planner(data, self.epsilon)


    def set_reward(self, reward):
        self.components.layer3.set_reward(reward)