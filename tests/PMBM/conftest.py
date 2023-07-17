import numpy as np
import pytest

from mot.common.gaussian_density import GaussianDensity as Gaussian


@pytest.fixture(scope="function")
def initial_PPP_intensity_linear():
    return Gaussian(np.array([-400.0, 200.0, 0.0, 0.0]), 400 * np.eye(4), np.log(0.03)) + Gaussian(
        np.array([-400.0, 200.0, 0.0, 0.0]), 400 * np.eye(4), np.log(0.03)
    )


@pytest.fixture(scope="function")
def initial_PPP_intensity_nonlinear():
    return (
        Gaussian(np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), np.eye(5), -0.3861)
        + Gaussian(np.array([20.0, 20.0, -20.0, 0.0, np.pi / 90]), np.eye(5), -0.423)
        + Gaussian(np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), np.eye(5), -1.8164)
    )
