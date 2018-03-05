# -*- coding: utf-8 -*-

from nose.tools import eq_

from noh.core import Architecture, Circuit


class SeedWBA(Architecture):
    def __init__(self):
        super(SeedWBA, self).__init__(
            (('hip', 'sa'), 'bg'),
            (('amg', 'sa'), 'hip'),
            (('hip', 'bg'), 'pfc'),
            ('hip', 'amg'),
        )


def test_architecture():
    seedwba = SeedWBA()
    test = Circuit(
        (('hip', 'sa'), 'bg'),
        ('sa', 'hip'),
        (('hip', 'bg'), 'pfc'),
    )

    seedwba.add_circuits(test=test)

    add = lambda a, b: a + b
    inc = lambda x: x + 1

    test.implement(sa=inc, bg=add, hip=inc, pfc=add)
    eq_(seedwba.test(0), 5)
    eq_(seedwba.test(1), 8)
