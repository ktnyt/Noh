# -*- coding: utf-8 -*-

from nose.tools import eq_

from noh.core import Architecture


class SeedWBA(Architecture):
    def __init__(self):
        super(SeedWBA, self).__init__(['sa', 'amg', 'bg', 'hip', 'pfc'], [
            (('hip', 'sa'), 'bg'),
            (('amg', 'sa'), 'hip'),
            (('hip', 'bg'), 'pfc'),
            ('hip', 'amg'),
        ])


def test_architecture():
    seedwba = SeedWBA()
    test = seedwba.circuit(
        (('hip', 'sa'), 'bg'),
        ('sa', 'hip'),
        (('hip', 'bg'), 'pfc'),
    )

    add = lambda a, b: a + b
    inc = lambda x: x + 1

    test.implement(sa=inc, bg=add, hip=inc, pfc=add)
    eq_(test(0), 5)
    eq_(test(1), 8)
