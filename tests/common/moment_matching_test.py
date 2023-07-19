import numpy as np
from src.common.gaussian_density import GaussianDensity


TOL = 1e-4

import numpy as np
from src.common.gaussian_density import GaussianDensity as GaussianDensity


def test_moment_matching():
    # Create a list of Gaussian objects
    gaussian1 = GaussianDensity(means=np.array([1, 2]), covs=np.array([[1, 0], [0, 1]]))
    gaussian2 = GaussianDensity(means=np.array([3, 4]), covs=np.array([[2, 1], [1, 2]]))
    gaussian3 = GaussianDensity(means=np.array([5, 6]), covs=np.array([[3, 2], [2, 3]]))
    gaussian_arr = gaussian1 + gaussian2 + gaussian3

    # Create a list of weights in logarithmic scale
    log_weights = [-0.1, -100, -100]

    # Call the moment_matching method
    result = GaussianDensity.moment_matching(log_weights, gaussian_arr)

    # Calculate the expected mean and covariance
    expected_mean = np.array([1, 2])
    expected_covariance = np.array([[1, 0], [0, 1]])

    # Compare the result with the expected output
    assert np.allclose(result.means, expected_mean, rtol=1e-3)
    assert np.allclose(result.covs, expected_covariance, rtol=1e-3)
