import numpy as np
from dataclasses import dataclass


@dataclass
class Estimation:
    track_id: int
    state_estimation: np.ndarray
