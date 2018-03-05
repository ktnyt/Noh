# -*- coding: utf-8 -*-

from itertools import chain
from inspect import signature, Signature, Parameter


def is_valid(nodes, edges):
    for edge in edges:
        source, target = edge
        if source not in nodes or target not in nodes:
            return False
    return True


def is_subset(a, b):
    for v in b:
        if v not in a:
            return False
    return True


def is_cyclic(edges):
    nodes = list(set(chain.from_iterable(edges)))

    visited = {}
    recstack = {}
    lut = {}

    for node in nodes:
        visited[node] = False
        recstack[node] = False
        lut[node] = []

    for u, v in edges:
        lut[u].append(v)

    def check(node):
        visited[node] = True
        recstack[node] = True

        for neighbour in lut[node]:
            if not visited[neighbour]:
                if check(neighbour):
                    return True
            elif recstack[neighbour]:
                return True

        recstack[node] = False
        return False

    for node in nodes:
        if not visited[node]:
            if check(node):
                return True
    return False


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


class Circuit(object):
    def __init__(self, *connections):
        edges = connections_to_edges(connections)
        nodes = list(set(chain.from_iterable(edges)))

        if not is_valid(nodes, edges):
            raise Exception(
                'Given nodes and edges set do not form a valid graph')

        if is_cyclic(edges):
            raise Exception('Error in circuit generation: circuit is cyclic')

        self.nodes = nodes
        self.edges = edges
        self.funcs = {}

        for node in self.nodes:
            self.funcs[node] = None

    def __call__(self, *args):
        given = list(args)
        incoming = self.nodes[:]
        outgoing = self.nodes[:]
        nodes = self.nodes[:]
        edges = self.edges

        outputs = {}

        for edge in edges:
            source, target = edge
            if target in incoming:
                incoming.remove(target)
            if source in outgoing:
                outgoing.remove(source)

        while nodes:
            node = nodes.pop(0)
            sig = signature(self.funcs[node])

            deps = [s for (s, t) in edges if t == node]

            ok = True
            for dep in deps:
                if dep not in outputs:
                    ok = False

            if not ok:
                nodes.append(node)
            else:
                inputs = []

                diff = len(sig.parameters) - len(deps)

                if diff > 0:
                    for _ in range(diff):
                        inputs.append(given.pop(0))

                for dep in deps:
                    inputs.append(outputs[dep])

                outputs[node] = self.funcs[node](*inputs)

        if len(outgoing) == 1:
            return outputs[outgoing[0]]

        returns = []

        for node in outgoing:
            returns.append(outputs[node])

        return tuple(returns)

    def implement(self, **funcs):
        missing = []

        for node in self.nodes:
            if node not in funcs:
                missing.append(node)

        if missing:
            tmp = ['\'{}\''.format(node) for node in missing].join(', ')
            raise Exception(
                'Error in Circuit instantiation: missing functions for {}'.
                format(tmp))

        self.funcs = funcs

        return self


class Architecture(object):
    def __init__(self, *connections):
        edges = connections_to_edges(connections)
        nodes = list(set(chain.from_iterable(edges)))

        if not is_valid(nodes, edges):
            raise Exception(
                'Given nodes and edges set do not form a valid graph')

        self.nodes = nodes
        self.edges = edges

    def add_circuits(self, **circuits):
        for name in circuits:
            circuit = circuits[name]
            if not is_subset(self.edges, circuit.edges):
                raise Exception(
                    'Error in circuit generation: not a valid subgraph')

            self.__setattr__(name, circuit)

        return self
