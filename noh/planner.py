from noh.component import Component

class Planner(Component):
    def __init__(self, components):
        self.components = components

    def __call__(self, data):
        pass

    def train(self, data):
        pass


"""
class Plan(object):
    def __init__(self, components):
        self.components = components

    def __call__(self, data, *args, **kwargs):
        raise NotImplementedError("`__call__` must be explicitly overridden")

class Planner(Plan):
    def __init__(self, components, **plans):
        self.plans = {}
        for plan in plans:
            self.plans[plan] = plans[plan](components)

    def __call__(self, plan, data, *args, **kwargs):
        return self.plans[plan](data, *args, **kwargs)

class DefaultCall(Plan):
    def __call__(self, data, *args, **kwargs):
        for component in self.components:
            data = component(data)
        return data

class ReverseCall(Plan):
    def __call__(self, data, *args, **kwargs):
        for component in reversed(self.components):
            data = component(data)
        return data

def Reversible(components):
    return Planner(components, forward=DefaultCall, backward=ReverseCall)

class DefaultTrain(Plan):
    def __call__(self, data, *args, **kwargs):
        errors = []
        for component in self.components:
            errors.append(component.train(data))
            data = component.train(data)
        return errors

class Backprop(Plan):
    def __call__(self, data, *args, **kwargs):
        error = func(data, label)
        for component in reversed(self.components):
            component.train(data, backprop=True)
        return error

def Pretrainable(components):
    return Planner(components, pretrain=DefaultTrain, finetune=Backprop)
"""
