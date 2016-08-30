import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from noh import Circuit, PropRule, TrainRule, Planner
from noh.components import RLTrainable

from noh.activate_functions import sigmoid, softmax, relu
from noh.components import Reservoir, PGLayer, ModelICONIP2016
from noh.environments import EightArmRadialMaze


class PGNetProp(PropRule):

    def __init__(self, components):
        super(PGNetProp, self).__init__(components)

    def __call__(self, input_data):

        data = input_data
        data = self.components.layer0(data)
        data = self.components.layer1(data)
        return data


class PGNetTrain(TrainRule):
    def __call__(self):
        error = None
        for component in [self.components.layer1, self.components.layer0]:
            error = component.train(error=error)


class PGNetPlanner(Planner):

    def __init__(self, components):
        super(PGNetPlanner, self).__init__(
            components,
            PropRules={"default":PGNetProp},
            TrainRules={"default":PGNetTrain})

    def __call__(self, input_data):
        res = self.default_prop_rule(input_data)
        return res

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
                           is_return_id=True, is_argmax=is_argmax,
                           mbatch_size=mbatch_size, lr=lr, decay_rate=decay_rate, gamma=gamma,
                           activate=softmax, reward_reset_checker=reward_reset_checker, is_output=True))
        RLTrainable.__init__(self)

    def __call__(self, data):
        return self.planner(data)

    def set_reward(self, reward):
        self.components.layer1.set_reward(reward)

if __name__ == "__main__":

    n_visible = EightArmRadialMaze.n_stat
    n_hidden = 50
    n_output = EightArmRadialMaze.n_act
    n_reward = EightArmRadialMaze.n_reward

    reward_reset_checker = EightArmRadialMaze.gen_reward_reset_checker()

    stl = Reservoir(n_visible=n_visible, n_hidden=n_hidden, activate=sigmoid,
                    bind_prob_W=1., bind_prob_W_rec=0.3)
    output_layer = PGNet(structure=[n_hidden, n_hidden, n_output],
                         is_argmax=False, mbatch_size=10, lr=1e-3, decay_rate=0.9, gamma=0.0,
                         reward_reset_checker=reward_reset_checker)
    model = ModelICONIP2016(stl=stl, output_layer=output_layer)

    env = EightArmRadialMaze(model)
    f_rewards = open("eight_arm_radial_maze_rewards.dat", "w")
    while True:
        env.exec_episode()
        f_rewards.write(str(env.reward_history[-1]) + "\n")

    f_rewards.close()
