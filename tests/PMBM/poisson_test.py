import copy

import numpy as np
import pytest
from mot.common import (
    GaussianDensity,
    GaussianMixture,
)
from mot.measurement_models import (
    ConstantVelocityMeasurementModel,
)
from mot.trackers.multiple_object_trackers.PMBM.common import (
    PoissonRFS,
    StaticBirthModel,
)
from scipy.special import logsumexp



@pytest.fixture
def dt():
    return 1.0


@pytest.fixture
def clutter_intensity():
    return 0.7 / 100


def test_PPP_predict_linear_motion(
    initial_PPP_intensity_linear, dt, cv_motion_model, P_S
):
    # Set Poisson RFS
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)
    PPP.predict(motion_model, survival_probability, dt)

    # check multiply of weight in log domain
    PPP_ref_w = np.array(
        [
            current_weight + np.log(survival_probability)
            for current_weight in initial_PPP_intensity_linear.log_weights
        ]
    )

    PPP_ref_state_x = [
        GaussianDensity.predict(component.gaussian, cv_motion_model, dt).x
        for component in initial_PPP_intensity_linear.weighted_components
    ]

    PPP_ref_state_P = [
        GaussianDensity.predict(component.gaussian, cv_motion_model, dt).P
        for component in initial_PPP_intensity_linear.weighted_components
    ]

    np.testing.assert_allclose(
        sorted(PPP.intensity.log_weights), sorted(PPP_ref_w), rtol=0.1
    )
    np.testing.assert_allclose(
        [gaussian.x for gaussian in PPP.intensity.states], PPP_ref_state_x, rtol=0.01
    )
    np.testing.assert_allclose(
        [gaussian.P for gaussian in PPP.intensity.states], PPP_ref_state_P, rtol=0.02
    )


def test_PPP_adds_birth_components(birth_model_linear):

    # Set Poisson RFS
    PPP = PoissonRFS(initial_intensity=GaussianMixture([]))
    birth_model = StaticBirthModel(birth_model_config=birth_model_linear)
    PPP.birth(new_components=birth_model.get_born_objects_intensity())

    np.testing.assert_allclose(
        sorted(PPP.intensity.log_weights),
        sorted(birth_model_linear.log_weights),
        rtol=0.1,
    )
    np.testing.assert_allclose(
        [gaussian.x for gaussian in PPP.intensity.states],
        [gaussian.x for gaussian in birth_model_linear.states],
        rtol=0.01,
    )
    np.testing.assert_allclose(
        [gaussian.P for gaussian in PPP.intensity.states],
        [gaussian.P for gaussian in birth_model_linear.states],
        rtol=0.02,
    )
    assert PPP.intensity[0] == birth_model_linear[0]


def test_PPP_undetected_update(initial_PPP_intensity_linear, P_D):
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)

    PPP.undetected_update(P_D)

    PPP_weights_ref = np.array(
        [
            log_weight + np.log(1 - P_D)
            for log_weight in initial_PPP_intensity_linear.log_weights
        ]
    )
    np.testing.assert_almost_equal(
        PPP.intensity.log_weights,
        PPP_weights_ref,
        decimal=4,
    )


def test_PPP_detected_update(
    initial_PPP_intensity_linear, dt, P_D, clutter_intensity, cv_measurement_model
):
    # Set Poisson RFS
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)

    z = np.array([[-410.0, 200.0]])
    measurement_indices_in_PPP = [True, False]

    bern, likelihood = PPP.detected_update(
        z,
        cv_measurement_model,
        P_D,
        clutter_intensity,
    )

    gated_PPP_component_indices = [
        idx
        for idx, _ in enumerate(initial_PPP_intensity_linear)
        if measurement_indices_in_PPP[idx] is True
    ]

    updated_initial_intensity = copy.deepcopy(initial_PPP_intensity_linear)
    for idx, component in enumerate(updated_initial_intensity):
        if idx in gated_PPP_component_indices:
            component.gaussian = GaussianDensity.update(
                component.gaussian, z, cv_measurement_model
            )

    log_likelihoods_per_measurement = np.array(
        [
            np.log(P_D)
            + component.log_weight
            + GaussianDensity.predicted_loglikelihood(
                component.gaussian, z, cv_measurement_model
            ).item()
            for idx, component in enumerate(updated_initial_intensity)
            if idx in gated_PPP_component_indices
        ]
    )

    log_likelihood_detection_from_object = logsumexp(log_likelihoods_per_measurement)
    log_likelihood_detection_from_clutter = np.log(clutter_intensity)
    ref_probability_existence = np.exp(
        log_likelihood_detection_from_object
        - np.logaddexp(
            log_likelihood_detection_from_object,
            log_likelihood_detection_from_clutter,
        )
    )
    import pdb

    pdb.set_trace()

    np.testing.assert_almost_equal(
        bern.existence_probability,
        ref_probability_existence,
        decimal=4,
    )

    referenced_likelihood = np.logaddexp(
        log_likelihood_detection_from_object, log_likelihood_detection_from_clutter
    )

    np.testing.assert_almost_equal(
        likelihood,
        referenced_likelihood,
        decimal=4,
    )
    assert PPP.intensity[0].gaussian == updated_initial_intensity[0].gaussian


def test_PPP_gating(initial_PPP_intensity_linear):
    # Create linear measurement model
    sigma_r = 5.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)

    z = np.array([[-410.0, 201.0], [10e6, 10e6]])
    gating_matrix_ud, meas_indices_ud = PPP.gating(
        z, GaussianDensity, meas_model, gating_size=0.99
    )
    gating_matrix_ud_ref = np.array([[True, False], [False, False]])

    meas_indices_ud_ref = np.array([True, False])

    np.testing.assert_almost_equal(
        gating_matrix_ud_ref,
        gating_matrix_ud,
        decimal=4,
    )

    np.testing.assert_almost_equal(
        meas_indices_ud_ref,
        meas_indices_ud,
        decimal=4,
    )


def test_PPP_pruning(initial_PPP_intensity_linear):
    modified_PPP_intensity = copy.deepcopy(initial_PPP_intensity_linear)
    modified_PPP_intensity.weighted_components[0].log_weight = -6
    PPP = PoissonRFS(initial_intensity=modified_PPP_intensity)
    PPP.prune(threshold=np.log(0.01))
    assert len(PPP) == 1
