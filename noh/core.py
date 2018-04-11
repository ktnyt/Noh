from uuid import uuid4
from abc import ABCMeta, abstractmethod


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

    def get(self, key):
        return self.buffer[key]

    def set(self, key, value):
        self.buffer[key] = value


class Sensor(Component):
    def __init__(self):
        super(Sensor, self).__init__()
        self.buffer = None

    def get(self, _):
        return self.buffer

    def set(self, _, value):
        self.buffer = value

    def input(self, value):
        self.buffer = value


class Actuator(Component, metaclass=ABCMeta):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Actuator, self).__init__()

    @abstractmethod
    def output(self):
        raise NotImplementedError()


class Circuit(object):
    def __init__(self, name, edges):
        self.name = name
        self.edges = edges
        self.funcs = [None] * len(edges)

    def __call__(self, arg):
        if None in self.funcs:
            raise NotImplementedError()

        if arg is not None:
            self.edges[0][0].set(self.name, arg)

        for (pre, post), func in zip(self.edges, self.funcs):
            post.set(self.name, func(pre.get(self.name)))

        return post.get(self.name)

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

        self.inputs = list(self.nodes.keys())
        self.outputs = list(self.nodes.keys())

        for key in self.edges:
            pre, post = key

            if pre not in components:
                raise Exception('no component with name {}'.format(pre))

            if post not in components:
                raise Exception('no component with name {}'.format(post))

            if pre in self.outputs:
                self.outputs.remove(pre)

            if post in self.inputs:
                self.inputs.remove(post)

        for key in self.inputs:
            if not isinstance(self.nodes[key], Sensor):
                raise Exception('input {} is not a Sensor'.format(key))

        for key in self.outputs:
            if not isinstance(self.nodes[key], Actuator):
                raise Exception('output {} is not an Actuator'.format(key))

    def add_circuit(self, name, circuit):
        if name in self.circuits:
            raise Warning('a circuit with name {} already exists'.format(name))
        self.circuits[name] = circuit

    def create_circuit(self, name, keys):
        if keys[0] not in self.inputs:
            raise Exception('{} is not an input node'.format(keys[0]))

        if keys[-1] not in self.outputs:
            raise Exception('{} is not an output node'.format(keys[-1]))

        wanted = zip(keys[:-1], keys[1:])
        edges = []

        for edge in wanted:
            if edge not in self.edges:
                raise Exception('{} is not a valid connection'.format(edge))
            pre, post = edge
            edges.append((self.nodes[pre], self.nodes[post]))

        circuit = Circuit(name, edges)
        self.add_circuit(name, circuit)
        return circuit

    def run_circuit(self, name, arg):
        return self.circuits[name](arg)

    def __call__(self, **args):
        for key in self.inputs:
            if key not in args:
                raise Exception('required parameter: {}', key)

        for key, value in args.items():
            if key not in self.inputs:
                raise Exception('invalid parameter: {}', key)
            self.nodes[key].input(value)

        for circuit in self.circuits.values():
            circuit(None)

        return {key: self.nodes[key].output() for key in self.outputs}
