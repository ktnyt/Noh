from noh.component import Component
from noh.wrapper import Wrapper

class Circuit(Component):
    def __init__(self, *components):
        self.components = components

    def __call__(self, data, **kwargs):
        for component in self.components:
            if len(kwargs.keys()) and isinstance(component, Wrapper):
                data = component(data, **kwargs)
            else:
                data = component(data)
        return data

    def train(self, data, **kwargs):
        errors = []
        for component in self.components:
            if len(kwargs.keys()) and isinstance(component, Wrapper):
                errors.append(component.train(data, **kwargs))
            else:
                errors.append(component.train(data))
            if len(kwargs.keys()) and isinstance(component, Wrapper):
                data = component(data, **kwargs)
            else:
                data = component(data)
        return errors
