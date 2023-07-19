from copy import deepcopy

import numpy as np
import pytest
from src.common.gaussian_density import GaussianDensity
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel

from src.trackers.multiple_object_trackers.PMBM.common.bernoulli import Bernoulli


@pytest.fixture
def initial_bernoulli():
    return Bernoulli(
        existence_probability=0.6,
        state=GaussianDensity(means=np.array([0.0, 0.0, 10.0, 10.0]), covs=np.eye(4)),
    )


@pytest.fixture
def cv_motion_model():
    dt = 1.0
    sigma_q = 5.0
    return ConstantVelocityMotionModel(dt, sigma_q)


@pytest.fixture
def cv_measurement_model():
    # Create linear measurement model
    sigma_r = 10.0
    return ConstantVelocityMeasurementModel(sigma_r)


@pytest.fixture
def P_S():
    """probability of survival"""
    return 0.8


@pytest.fixture
def P_D():
    """probability of detection"""
    return 0.7


@pytest.fixture
def outlier_measurement():
    return np.array([[100.0, 100.0]])


@pytest.fixture
def neighbour_measurement():
    return np.array([[1.0, 1.0]])


def test_bern_predict(initial_bernoulli: Bernoulli, cv_motion_model, P_S):
    # Create nonlinear motion model (coordinate turn)
    initial_bernoulli.predict(cv_motion_model, P_S, GaussianDensity, 1.0)
    # reference
    r_ref = 0.48
    state_ref = GaussianDensity(
        means=np.array([10.0, 10.0, 10.0, 10.0]),
        covs=np.array(
            [
                [8.2500, 0, 13.5000, 0],
                [0, 8.2500, 0, 13.5000],
                [13.5000, 0, 26.0000, 0],
                [0, 13.5000, 0, 26.0000],
            ]
        ),
    )
    np.testing.assert_allclose(
        r_ref, initial_bernoulli.existence_probability, rtol=0.05
    )
    np.testing.assert_allclose(
        state_ref.means, initial_bernoulli.state.means, atol=1e-4
    )
    np.testing.assert_allclose(state_ref.covs, initial_bernoulli.state.covs, atol=1e-4)


def test_bern_undetected_update(initial_bernoulli, P_D):
    bern = deepcopy(initial_bernoulli)
    ref_bern_r = 0.31
    ref_lik_undetected = -0.54

    new_bern = bern.undetected_update_state(P_D)
    log_likelihood_undetected = bern.undetected_update_loglikelihood(P_D)

    np.testing.assert_allclose(new_bern.existence_probability, ref_bern_r, rtol=0.05)
    np.testing.assert_allclose(log_likelihood_undetected, ref_lik_undetected, rtol=0.05)


def test_bern_detected_update_likelihood_outlier(
    initial_bernoulli, P_D, cv_measurement_model, outlier_measurement
):
    likelihood_detected = initial_bernoulli.detected_update_loglikelihood(
        outlier_measurement, cv_measurement_model, P_D
    )
    likelihood_detected_ref = np.array([-105.7914])
    np.testing.assert_allclose(likelihood_detected, likelihood_detected_ref, rtol=0.05)


def test_bern_detected_update_likelihood_target(
    initial_bernoulli, P_D, cv_measurement_model, neighbour_measurement
):
    log_likelihood_detected = initial_bernoulli.detected_update_loglikelihood(
        neighbour_measurement, cv_measurement_model, P_D
    )

    log_likelihood_detected_ref = np.array([-7.3304])
    np.testing.assert_allclose(
        log_likelihood_detected, log_likelihood_detected_ref, rtol=0.05
    )


def test_bern_update_state(
    initial_bernoulli, neighbour_measurement, cv_measurement_model
):
    # Perform the update
    new_bernoulli = Bernoulli.detected_update_state(
        initial_bernoulli, neighbour_measurement, cv_measurement_model
    )

    # Check the existence probability
    ref_r = 1.0
    np.testing.assert_allclose(new_bernoulli.existence_probability, ref_r, rtol=0.01)

    # Check the state
    ref_state_x = np.array([[0.0099, 0.0099, 10.0, 10.0]])
    np.testing.assert_allclose(new_bernoulli.state.means, ref_state_x, rtol=0.01)

    # Check the covariance
    ref_state_P = np.array(
        [
            [
                [0.9901, 0.0, 0.0, 0.0],
                [0.0, 0.9901, 0.0, 0.0],
                [0.0, 0.0, 1.0000, 0.0],
                [0.0, 0.0, 0.0, 1.0000],
            ]
        ]
    )
    np.testing.assert_allclose(new_bernoulli.state.covs, ref_state_P, rtol=0.01)
