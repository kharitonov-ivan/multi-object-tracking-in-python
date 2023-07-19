import numpy as np
from src.common.gaussian_density import GaussianDensity


birth_model_params = (
    GaussianDensity(np.array([0.0, 0.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03))
    + GaussianDensity(
        np.array([400.0, -600.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03)
    )
    + GaussianDensity(
        np.array([-800.0, 200.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03)
    )
    + GaussianDensity(
        np.array([-200.0, 800.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03)
    )
)
