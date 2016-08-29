from noh.utils import Collection

class PropRule(Collection):
    def __init__(self, components):
        super(PropRule, self).__init__(components)

    def __call__(self, data):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class TrainRule(Collection):
    def __init__(self, components):
        super(TrainRule, self).__init__(components)

    def __call__(self, data, label, epoch):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Planner(object):
    def __init__(self, prop, train, components, **Rules):
        self.components = components
        self.rules = {
            'prop': prop(components),
            'train': train(components)
        }

        for name in Rules:
            Rule = Rules[name]
            self.rules[name] = Rule(components, labels)

        self.prop_rule = self.rules['prop']
        self.train_rule = self.rules['train']

    def set_prop(self, name):
        self.prop_rule = self.rules[name]

    def set_train(self, name):
        self.train_rule = self.rules[name]

    def __call__(self, data):
        return self.prop_rule(data)

    def train(self, data, label, epoch):
        return self.train_rule(data, label, epoch)
