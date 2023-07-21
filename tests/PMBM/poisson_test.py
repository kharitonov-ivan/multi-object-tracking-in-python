import copy

import numpy as np
import pytest

from src.common.gaussian_density import GaussianDensity
from src.configs import SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.trackers.multiple_object_trackers.PMBM.common import (
    PoissonRFS,
    StaticBirthModel,
)
from tests.PMBM.params.birth_model import birth_model_params


@pytest.fixture
def dt():
    return 1.0


@pytest.fixture
def clutter_intensity():
    return 0.7 / 100


def test_PPP_predict_linear_motion(initial_PPP_intensity_linear, clutter_intensity):
    survival_probability = 0.9

    dt = 1.0
    sigma_q = 10.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Set Poisson RFS
    PPP = PoissonRFS(intensity=initial_PPP_intensity_linear)
    PPP.predict(motion_model, survival_probability, GaussianDensity, dt)

    # check multiply of weight in log domain
    PPP_ref_w = np.array([current_weight + np.log(survival_probability) for current_weight in initial_PPP_intensity_linear.weights])

    PPP_ref_state = GaussianDensity.predict(initial_PPP_intensity_linear, motion_model, dt)
    np.testing.assert_allclose(sorted(PPP.intensity.weights), sorted(PPP_ref_w), rtol=0.1)
    np.testing.assert_allclose(PPP.intensity.means, PPP_ref_state.means, rtol=0.01)
    np.testing.assert_allclose(PPP.intensity.covs, PPP_ref_state.covs, rtol=0.02)


def test_PPP_adds_birth_components():
    # Set Poisson RFS
    PPP = PoissonRFS(intensity=GaussianDensity())
    birth_model = StaticBirthModel(birth_model_density=birth_model_params)
    PPP.birth(new_components=birth_model.get_born_objects_intensity())

    np.testing.assert_allclose(
        sorted(PPP.intensity.weights),
        sorted(birth_model_params.weights),
        rtol=0.1,
    )
    np.testing.assert_allclose(
        PPP.intensity.means,
        birth_model_params.means,
        rtol=0.01,
    )
    np.testing.assert_allclose(
        PPP.intensity.covs,
        birth_model_params.covs,
        rtol=0.02,
    )
    assert len(PPP.intensity) == len(birth_model_params)


def test_PPP_undetected_update(initial_PPP_intensity_linear):
    detection_probability = 0.8
    PPP = PoissonRFS(intensity=initial_PPP_intensity_linear)
    PPP.undetected_update(detection_probability)
    PPP_weights_ref = initial_PPP_intensity_linear.weights + np.log(1 - detection_probability)
    np.testing.assert_almost_equal(PPP.intensity.weights, PPP_weights_ref)


def test_PPP_detected_update(initial_PPP_intensity_linear):
    detection_probability = 0.8
    clutter_rate = 1.0  # aka lambda_c

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = SensorModelConfig(P_D=detection_probability, lambda_c=clutter_rate, range_c=range_c)
    clutter_intensity = sensor_model.intensity_c

    # Create nlinear motion model
    dt = 1.0
    sigma_q = 10.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)  # noqa F841

    # Create linear measurement model
    sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Set Poisson RFS
    PPP = PoissonRFS(intensity=copy.deepcopy(initial_PPP_intensity_linear))
    gating_size = 0.8
    measurements = np.array([[-410.0, 200.0]])
    measurement_indices_in_PPP = [True, True]

    new_sth = PPP.get_targets_detected_for_first_time(
        measurements,
        sensor_model.intensity_c,
        meas_model,
        detection_probability,
        gating_size,
    )

    # gated_PPP_component_indices = [idx for idx, _ in enumerate(initial_PPP_intensity_linear) if measurement_indices_in_PPP[idx] is True]

    # updated_initial_intensity = copy.deepcopy(initial_PPP_intensity_linear)
    # for idx, component in enumerate(updated_initial_intensity):
    #     if idx in gated_PPP_component_indices:
    #         component.gaussian = GaussianDensity.update(component, measurements, meas_model)

    # log_likelihoods = GaussianDensity.predict_loglikelihood(component, measurements, meas_model).item()
    #     for component in copy.deepcopy(initial_PPP_intensity_linear)
    # ]

    # log_likelihoods_per_measurement = np.array(
    #     [
    #         np.log(detection_probability) + component.weights + log_likelihood
    #         for idx, (component, log_likelihood) in enumerate(zip(updated_initial_intensity, log_likelihoods))
    #         if idx in gated_PPP_component_indices
    #     ]
    # )

    # log_likelihood_detection_from_object = logsumexp(log_likelihoods_per_measurement)

    # log_likelihood_detection_from_clutter = np.log(clutter_intensity)

    # ref_probability_existence = np.exp(
    #     log_likelihood_detection_from_object
    #     - np.logaddexp(
    #         log_likelihood_detection_from_object,
    #         log_likelihood_detection_from_clutter,
    #     )
    # )

    # np.testing.assert_almost_equal(
    #     new_sth.bernoulli.existence_probability,
    #     ref_probability_existence,
    #     decimal=4,
    # )

    # referenced_likelihood = np.logaddexp(log_likelihood_detection_from_object, log_likelihood_detection_from_clutter)

    # np.testing.assert_almost_equal(
    #     new_sth.log_likelihood,
    #     referenced_likelihood,
    #     decimal=4,
    # )

    # # TODO Do we have to update PPP intensity here or not?
    # try:
    #     np.testing.assert_almost_equal(PPP.intensity[0].gaussian.x, updated_initial_intensity[0].gaussian.x)
    # except AssertionError:
    #     logging.debug("They are different!")


def test_PPP_gating(initial_PPP_intensity_linear):
    # Create linear measurement model
    sigma_r = 5.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    PPP = PoissonRFS(intensity=initial_PPP_intensity_linear)

    z = np.array([[-410.0, 201.0], [10e6, 10e6]])
    gating_matrix_ud, dists = GaussianDensity.ellipsoidal_gating(PPP.intensity, z, meas_model, 0.8)
    gating_matrix_ud_ref = np.array([[True, False], [True, False]])
    np.testing.assert_almost_equal(
        gating_matrix_ud_ref,
        gating_matrix_ud,
        decimal=4,
    )


def test_PPP_pruning(initial_PPP_intensity_linear):
    initial_PPP_intensity_linear.weights[0] = -6
    PPP = PoissonRFS(intensity=initial_PPP_intensity_linear)
    PPP.prune(threshold=np.log(0.01))

    assert len(PPP.intensity) == 1
