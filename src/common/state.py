from dataclasses import dataclass

import numpy as np


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
