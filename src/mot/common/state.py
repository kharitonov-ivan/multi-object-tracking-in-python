import collections
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Gaussian:
    """Describes state of object

    Parameters
    ----------
    x : np.ndarray (N), N - state dimension
        describes state vector of object
    P : np.ndarray (N x N), N - state dimension
        describes covariance of state of object
    """

    x: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        assert isinstance(self.x, np.ndarray), "Argument of wrong type!"
        assert isinstance(self.P, np.ndarray), "Argument of wrong type!"
        assert self.x.ndim == 1, "x must be N - vector"
        assert self.P.ndim == 2, "P must be N x N matrix"
        assert self.P.shape[0] == self.P.shape[1], "Covariance matrix should be square!"
        assert (
            self.P.shape[0] == self.x.shape[0]
        ), "size of vector should be equal P column size!"

    def __repr__(self) -> str:
        np.set_printoptions(linewidth=np.inf)
        return (
            f"{self.__class__.__name__} "
            f"x = {np.array2string(self.x, max_line_width=np.inf, precision =1)}  "
        )


@dataclass
class WeightedGaussian:
    w: float
    gm: Gaussian

    def __post_init__(self):
        assert not np.isnan(self.w), "nan weight"

    def __repr__(self) -> str:
        return (
            f"\n"
            f"weighted Gaussian with: "
            f"w = {self.w:.2f}   "
            f"x = {np.array2string(self.gm.x, max_line_width=np.inf, precision =2)}  "
        )


class GaussianMixture(collections.MutableSequence):
    def __init__(self, weighted_components: List[WeightedGaussian] = (None)):
        self.weighted_components = weighted_components

    @property
    def weights(self):
        weights = [x.w for x in self.weighted_components]
        return weights

    @weights.setter
    def weights(self, weights_value):
        assert len(weights_value) == len(self.weighted_components)
        for idx in range(len(self.weighted_components)):
            self.weighted_components[idx].w = weights_value[idx]

    @property
    def size(self):
        return len(self.weighted_components)

    @property
    def states(self):
        states = [x.gm for x in self.weighted_components]
        return states

    def check(self, item):
        if not isinstance(item, (WeightedGaussian)):
            raise TypeError

    def __len__(self):
        return len(self.weighted_components)

    def __getitem__(self, idx):
        return self.weighted_components[idx]

    def __delitem__(self, idx):
        del self.weighted_components[idx]

    def __setitem__(self, idx, item):
        self.check(item)
        self.weighted_components[idx] = item

    def insert(self, idx, item):
        self.check(item)
        self.weighted_components.insert(idx, item)

    def append(self, other):
        assert isinstance(other, WeightedGaussian)
        self.weighted_components.append(other)

    def extend(self, other):
        assert isinstance(other, GaussianMixture)
        for idx in range(len(other)):
            self.append(other[idx])

    def __repr__(self) -> str:
        return f"Gaussian Mixture " f"components={self.weighted_components}"
