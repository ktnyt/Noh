# -*- coding: utf-8 -*-
import random
from nose.tools import eq_

from noh.core import Architecture, Circuit, Component, Sensor, Actuator


class PFC(Actuator):
    def __init__(self):
        super(PFC, self).__init__()

    def output(self):
        keys =list(self.buffer.keys())
        return self.buffer[random.choice(keys)]


class SeedWBA(Architecture):
    def __init__(self):
        super(SeedWBA, self).__init__(
            dict(
                sa=Sensor(),
                hip=Component(),
                amg=Component(),
                bg=Component(),
                pfc=PFC(),
            ), [
                (('hip', 'sa'), 'bg'),
                (('amg', 'sa'), 'hip'),
                ('hip', 'amg'),
                (('hip', 'bg'), 'pfc'),
            ])


def test_architecture():
    seedwba = SeedWBA()
    test = seedwba.create_circuit('test', ['sa', 'bg', 'pfc'])
    double = lambda x: x * 2
    test.implement(double, double)

    eq_(seedwba.run_circuit('test', 42), 168)
    eq_(seedwba(sa=42), {'pfc': 168})
