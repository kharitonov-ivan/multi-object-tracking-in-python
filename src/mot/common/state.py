import collections
from collections import UserList
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Any
from nptyping import NDArray, Shape, Float
import numpy as np
from regex import F


@dataclass
class ObjectMetadata:
    object_class: str
    confidence: float
    size: np.ndarray


class EstimationObjectMetadata(ObjectMetadata):
    track_id: int


@dataclass
class Observation:
    measurement: np.ndarray
    metadata: ObjectMetadata


@dataclass
class Estimation:
    state: np.ndarray
    covariance: np.ndarray
    metadata: EstimationObjectMetadata



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
        assert self.P.shape[0] == self.x.shape[0], "size of vector should be equal P column size!"

    def __repr__(self) -> str:
        np.set_printoptions(linewidth=np.inf)
        return f"{self.__class__.__name__} " f"x = {np.array2string(self.x, max_line_width=np.inf, precision =1)}"

    def __eq__(self, other) -> bool:
        assert isinstance(other, Gaussian)
        if self.x.__eq__(other.x).all() and self.P.__eq__(other.P).all():
            return True
        else:
            return False

    @property
    def states_np(self):
        return self.x

    @property
    def covariances_np(self):
        return self.P


@dataclass
class WeightedGaussian:
    log_weight: float
    gaussian: Gaussian

    def __post_init__(self):
        assert not np.isnan(self.log_weight), "nan weight"

    def __repr__(self) -> str:
        return (
            f"\n"
            f"weighted Gaussian with: "
            f"w = {self.log_weight:.2f}   "
            f"x = {np.array2string(self.gaussian.x, max_line_width=np.inf, precision =2)}  "
        )

@dataclass
class GaussianMixtureNumpy:
    means: NDArray[Shape["Batch_size, State_dim"], Float]
    covs: NDArray[Shape["Batch_size, State_dim, State_dim"], Float]
    weigths_log: NDArray[Shape["Batch_size"], Float]

    def __post_init__(self):
        if len(self.means.shape) == 1: self.means = np.expand_dims(self.means, axis=0)
        if len(self.covs.shape) == 2: self.covs = np.expand_dims(self.covs, axis=0)
        if len(self.weigths_log.shape) == 0: self.weigths_log = np.array([self.weigths_log])

        assert self.means.shape[0] == self.covs.shape[0] == self.weigths_log.shape[0], "Number of components must be equal"
        assert len(self.means.shape) == 2
        assert len(self.covs.shape) == 3
        assert len(self.weigths_log.shape) == 1

    @classmethod
    def from_gaussian_mixture(cls, gaussian_mixture):
        return cls(
            means=gaussian_mixture.states_np,
            covs=gaussian_mixture.covariances_np,
            weigths_log=gaussian_mixture.log_weights,
        )
    
    def __add__(self, other):
        return GaussianMixtureNumpy(
            means=np.concatenate((self.means, other.means)),
            covs=np.concatenate((self.covs, other.covs)),
            weigths_log=np.concatenate((self.weigths_log, other.weigths_log)),
        )

    @property
    def size(self):
        return self.means.shape[0]

    def __len__(self):
        return self.size
    
    def extend(self, other):
        self.means = np.concatenate((self.means, other.means))
        self.covs = np.concatenate((self.covs, other.covs))
        self.weigths_log = np.concatenate((self.weigths_log, other.weigths_log))

class GaussianMixture(UserList):
    def __init__(self, initial_components: List[WeightedGaussian] = None):
        self.data = initial_components

    @property
    def log_weights(self):
        if self.data:
            weights = [x.log_weight for x in self.data]
            return weights
        else:
            return None

    @log_weights.setter
    def log_weights(self, log_weights):
        assert len(log_weights) == len(self.data)
        for idx in range(len(self.data)):
            self.weighted_components[idx].log_weights = log_weights[idx]

    @property
    def size(self):
        return len(self.data)

    @property
    def states(self):
        return [x.gaussian for x in self.data]

    @property
    def states_np(self):
        return np.array([state.gaussian.x for state in self.data])

    @property
    def covariances_np(self):
        return np.array([state.gaussian.P for state in self.data])


class _GaussianMixture(collections.abc.MutableSequence):
    def __init__(self, weighted_components: List[WeightedGaussian] = (None)):
        self.weighted_components = deepcopy(weighted_components)

    @property
    def log_weights(self):
        if self.weighted_components:
            weights = [x.log_weight for x in self.weighted_components]
            return weights
        else:
            return None

    @log_weights.setter
    def log_weights(self, log_weights):
        assert len(log_weights) == len(self.weighted_components)
        for idx in range(len(self.weighted_components)):
            self.weighted_components[idx].log_weights = log_weights[idx]

    @property
    def size(self):
        return len(self.weighted_components)

    @property
    def states(self):
        return [x.gaussian for x in self.weighted_components]

    @property
    def states_np(self):
        return np.array([state.gaussian.x for state in self.weighted_components])

    @property
    def covariances_np(self):
        return np.array([state.gaussian.P for state in self.weighted_components])

    def __copy__(self):
        return GaussianMixture(weighted_components=self.weighted_components)

    def check(self, item):
        if not isinstance(item, (WeightedGaussian)):
            raise TypeError

    def __len__(self):
        if self.weighted_components:
            return len(self.weighted_components)
        else:
            return 0

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
        if not self.weighted_components:
            self.weighted_components = []
        self.weighted_components.append(other)

    def extend(self, other):
        assert isinstance(other, GaussianMixture)
        for idx in range(len(other)):
            self.append(other[idx])

    def __repr__(self) -> str:
        return f"Gaussian Mixture " f"components={self.weighted_components}"


class NumpyGaussian(np.recarray):
    def __new__(cls, mean, covariance, ndim):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        dtype = np.dtype([("means", np.float32, (ndim,)), ("covariances", np.float32, (ndim, ndim))])
        obj = np.asarray((mean, covariance), dtype=dtype).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, "info", None)


class NumpyWeightedGaussian(np.recarray):
    def __new__(cls, log_weight, gaussian):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        dtype = np.dtype([("log_weight", log_weight.dtype), ("gaussian", gaussian.dtype)])
        obj = np.asarray((log_weight, gaussian), dtype=dtype).view(cls)
        # Finally, we must return the newly created object:
        return obj


if __name__ == "__main__":
    a = Gaussian(mean=[2.0, 2.0, 3.0], covariance=np.eye(3), ndim=3)
    b = WeightedGaussian(log_weight=0.03)