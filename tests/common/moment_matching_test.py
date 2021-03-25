import numpy as np
import scipy
import scipy.io
from mot.common.gaussian_density import GaussianDensity
from mot.common.state import State

TOL = 1e-4


def test_moment_matching():
    num_gaussians = 5
    n_dim = 4
    states = []
    expected_vars = scipy.io.loadmat("tests/data/SA2Ex2Test3.mat")

    test_w = np.array(expected_vars["w"]).squeeze()
    test_states = [
        State(x=np.array(test_x[0]).squeeze(), P=np.array(test_P)[0])
        for test_x, test_P in zip(
            expected_vars["states"]["x"], expected_vars["states"]["P"]
        )
    ]
    got_state = GaussianDensity.moment_matching(test_w, test_states)
    expected_state = State(
        x=expected_vars["state_ref"]["x"][0][0].squeeze(),
        P=expected_vars["state_ref"]["P"][0][0],
    )
    assert np.linalg.norm(got_state.x).all() > TOL, "check calculation of mean"
    assert (
        np.linalg.norm(got_state.P - expected_state.P).all() > TOL
    ), "check calculation of covariance"
