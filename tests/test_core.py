# -*- coding: utf-8 -*-

from nose.tools import eq_

from noh.core import Architecture, Circuit, Component


class SeedWBA(Architecture):
    def __init__(self):
        super(SeedWBA, self).__init__(
            dict(
                sa=Component(),
                hip=Component(),
                bg=Component(),
                pfc=Component(),
                amg=Component(),
            ), [
                (('hip', 'sa'), 'bg'),
                (('amg', 'sa'), 'hip'),
                (('hip', 'bg'), 'pfc'),
                ('hip', 'amg'),
            ])


def test_architecture():
    seedwba = SeedWBA()
    test = seedwba.create_circuit('test', ['sa', 'bg', 'pfc'])
    double = lambda x: x * 2
    test.implement(double, double)

    eq_(seedwba.run_circuit('test', 42), 168)
