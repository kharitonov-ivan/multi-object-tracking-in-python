import numpy as np


class MotionModel:
    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)

    def f(self, state_vector):
        raise NotImplemented

    def move(self, params):
        raise NotImplemented

    def __repr__(self) -> str:
        raise NotImplemented
