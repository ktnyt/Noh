from noh.component import Component
from noh.utils import Collection

class Circuit(Collection, Component):
    def __init__(self, planner, components):
        super(Circuit, self).__init__(components)
        self.planner = planner(components)

    def __call__(self, data):
        return self.planner(data)

    def train(self, data, label, epochs):
        return self.planner.train(data, label, epochs)
        
