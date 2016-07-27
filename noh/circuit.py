from noh.component import Component

class Circuit(Component):
    def __init__(self, components, planner):
        self.components = components
        self.planner = planner

    def __call__(self, data):
        return self.planner(data)

    def train(self, data, label, epochs):
        return self.planner.train(data, label, epochs)
