import numpy as np

from src.common.gaussian_density import GaussianDensity
from src.common.state import Gaussian


def test_moment_matching():
    # Setup data
    weights = np.log(np.array([0.3, 0.7], dtype=float))  # now in log form and float
    states = [
        Gaussian(x=np.array([1, 2], dtype=float), P=np.array([[1, 0], [0, 1]], dtype=float)),
        Gaussian(x=np.array([3, 4], dtype=float), P=np.array([[2, 1], [1, 2]], dtype=float)),
    ]

    # Run the method
    result = GaussianDensity.moment_matching(weights, states)

    # Check the result's mean
    log_weights_exp = np.exp(weights)
    expected_mean = log_weights_exp[0] * states[0].x + log_weights_exp[1] * states[1].x
    np.testing.assert_array_almost_equal(result.x, expected_mean)

    # Check the result's covariance
    expected_cov = np.zeros_like(states[0].P, dtype=float)
    for idx in range(len(log_weights_exp)):
        d = result.x - states[idx].x
        expected_cov += (states[idx].P + d @ d.T) * log_weights_exp[idx]
    np.testing.assert_array_almost_equal(result.P, expected_cov)
