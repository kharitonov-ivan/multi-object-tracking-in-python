import numpy as np
import pytest

from mot.common.gaussian_density import GaussianDensity


def test_create_state():
    state = GaussianDensity(means=np.array([[0, 0]]), covs=np.eye(2)[None, ...])
    np.testing.assert_allclose(state.means, np.array([[0, 0]]))
    np.testing.assert_allclose(state.covs, np.eye(2)[None, ...])
