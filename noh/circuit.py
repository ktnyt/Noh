from noh.component import Component

class Circuit(Component):
    def __init__(self, components):
        self.components = components

    def __call__(self, data):
        for component in self.components:
            data = component(data)
        return data

    def train(self, data):
        errors = []
        for component in self.components:
            errors.append(component.train(data))
            data = component.train(data)
        return errors
