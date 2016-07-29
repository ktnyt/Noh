from noh.component import Component
from noh.utils import DotAccessible

class Circuit(Component):
    def __init__(self, router, **components):
        self.components = DotAccessible(components)
        self.router = router
        self.router.setup(components)

    def __call__(self, data):
        return self.router(data)

    def train(self, data, label, epochs):
        return self.router.train(data, label, epochs)
