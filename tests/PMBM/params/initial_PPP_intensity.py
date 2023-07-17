import numpy as np
import pytest

from mot.common import Gaussian, GaussianMixture, WeightedGaussian


@pytest.fixture(scope="function")
def initial_PPP_intensity_linear():
    return GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(means=np.array([-400.0, 200.0, 0.0, 0.0]), covs=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(means=np.array([-400.0, -200.0, 0.0, 0.0]), covs=400 * np.eye(4)),
            ),
        ]
    )


@pytest.fixture(scope="function")
def initial_PPP_intensity_nonlinear():
    return GaussianMixture(
        [
            WeightedGaussian(
                -0.3861,
                Gaussian(means=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), covs=np.eye(5)),
            ),
            WeightedGaussian(
                -0.423,
                Gaussian(means=np.array([20.0, 20.0, -20.0, 0.0, np.pi / 90]), covs=np.eye(5)),
            ),
            WeightedGaussian(
                -1.8164,
                Gaussian(means=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), covs=np.eye(5)),
            ),
        ]
    )
