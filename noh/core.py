from uuid import uuid4
from inspect import signature, Signature, Parameter


def connections_to_edges(connections):
    edges = []
    for connection in connections:
        sources, target = connection

        if type(sources) not in [tuple, list]:
            sources = connection[:-1]
            target = connection[-1]

        for source in sources:
            edges.append((source, target))

    return edges


class Component(object):
    def __init__(self):
        self.buffer = {}

    def get(self, name):
        return self.buffer[name]

    def set(self, name, value):
        self.buffer[name] = value


class Circuit(object):
    def __init__(self, edges):
        self.uuid = uuid4()
        self.edges = edges
        self.funcs = [None] * len(edges)

    def __call__(self, arg):
        if None in self.funcs:
            raise NotImplementedError()

        for (pre, post), func in zip(self.edges, self.funcs):
            pre.set(self.uuid, arg)
            arg = func(arg)
            post.set(self.uuid, arg)

        return arg

    def implement(self, *funcs):
        if len(funcs) != len(self.funcs):
            wanted = len(self.funcs)
            given = len(funcs)
            raise Exception('wanted {} function(s) got but {}'.format(
                wanted, given))
        self.funcs = funcs


class Architecture(Component):
    def __init__(self, components, connections):
        super(Architecture, self).__init__()
        self.edges = connections_to_edges(connections)
        self.nodes = components
        self.circuits = {}

        for key in self.edges:
            pre, post = key

            if pre not in components:
                raise Exception('no component with name {}'.format(pre))

            if post not in components:
                raise Exception('no component with name {}'.format(post))

    def add_circuit(self, name, circuit):
        if name in self.circuits:
            raise Warning('a circuit with name {} already exists'.format(name))
        self.circuits[name] = circuit

    def create_circuit(self, name, keys):
        wanted = zip(keys[:-1], keys[1:])
        edges = []

        for edge in wanted:
            if edge not in self.edges:
                raise Exception('{} is not a valid connection'.format(edge))
            pre, post = edge
            edges.append((self.nodes[pre], self.nodes[post]))

        circuit = Circuit(edges)
        self.add_circuit(name, circuit)
        return circuit

    def run_circuit(self, name, arg):
        return self.circuits[name](arg)
