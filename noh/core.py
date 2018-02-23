# -*- coding: utf-8 -*-
"""Defines the core classes for Noh.

The core module provides the core classes and functions for Noh. These include
``Component``, ``Circuit``, ``Environment``, and ``asComponent``.

"""

from abc import ABCMeta, abstractmethod
from enum import Enum


def as_component(f):
    return FunctionComponent(f)


class TrainMode(Enum):
    Supervised = '@@NOH/TRAIN_MODE/SUPERVISED'
    Unsupervised = '@@NOH/TRAIN_MODE/UNSUPERVISED'
    Reinforcement = '@@NOH/TRAIN_MODE/REINFORCEMENT'


class Component(object, metaclass=ABCMeta):
    def __init__(self):
        self.train_mode = False

    def __call__(self, *args):
        if self.train_mode == TrainMode.Supervised:
            self.supervised(*args)
        elif self.train_mode == TrainMode.Unupervised:
            self.unsupervised(*args)
        elif self.train_mode == TrainMode.Reinforcement:
            self.reinforcement(*args)

        return self.execute(args)

    @abstractmethod
    def execute(self, *args):
        raise NotImplementedError()

    def supervised(self, inputs, labels):
        raise NotImplementedError()

    def unsupervised(self, inputs):
        raise NotImplementedError()

    def reinforcement(self, state, reward, action):
        raise NotImplementedError()


class FunctionComponent(object):
    def __init__(self, f):
        super(FunctionComponent, self).__init__()
        self.f = f

    def execute(self, *args):
        return self.f(*args)
