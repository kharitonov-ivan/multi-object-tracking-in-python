import numpy as np


class MeasurementModel:
    """
    MeasurementModel is a abstract class for different measurement models.
    """

    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)

    def observe(self, params):
        raise NotImplemented

    @property
    def dim(self):
        raise NotImplemented
