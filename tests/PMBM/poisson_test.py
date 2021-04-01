import numpy as np

from pytest import fixture

from mot.common import Gaussian, GaussianMixture, WeightedGaussian, GaussianDensity

from mot.measurement_models import (
    ConstantVelocityMeasurementModel,
    RangeBearingMeasurementModel,
)
from mot.motion_models import CoordinateTurnMotionModel, ConstantVelocityMotionModel
from mot.trackers.multiple_object_trackers.PMBM.common import PoissonRFS
from .params.birth_model import birth_model_linear
from .params.initial_PPP_intensity import initial_PPP_intensity_linear
import copy


def test_PPP_predict_linear_motion(initial_PPP_intensity_linear):
    # constant timestep
    dt = 1.0

    # Create linear motion model
    sigma_q = 1.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Set probability of esitense
    survival_probability = 0.8

    # Set Poisson RFS
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)
    PPP.predict(motion_model, survival_probability, dt)

    # check multiply of weight in log domain
    PPP_ref_w = np.array([
        current_weight + np.log(survival_probability)
        for current_weight in initial_PPP_intensity_linear.log_weights
    ])

    PPP_ref_state_x = [
        GaussianDensity.predict(component.gaussian, motion_model, dt).x
        for component in initial_PPP_intensity_linear.weighted_components
    ]

    PPP_ref_state_P = [
        GaussianDensity.predict(component.gaussian, motion_model, dt).P
        for component in initial_PPP_intensity_linear.weighted_components
    ]

    np.testing.assert_allclose(sorted(PPP.intensity.log_weights),
                               sorted(PPP_ref_w),
                               rtol=0.1)
    np.testing.assert_allclose(
        [gaussian.x for gaussian in PPP.intensity.states],
        PPP_ref_state_x,
        rtol=0.01)
    np.testing.assert_allclose(
        [gaussian.P for gaussian in PPP.intensity.states],
        PPP_ref_state_P,
        rtol=0.02)


def test_PPP_detected_update(initial_PPP_intensity_nonlinear):
    P_D = 0.8
    clutter_intensity = 0.7 / 100

    # Create nonlinear measurement model (range/bearing)
    sigma_r = 5.0
    sigma_b = np.pi / 180
    s = np.array([300, 400]).T
    meas_model = RangeBearingMeasurementModel(sigma_r=sigma_r,
                                              sigma_b=sigma_b,
                                              sensor_pos=s)

    # Set Poisson RFS
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_nonlinear)

    indices = [True, False, True]
    z = np.array([[0.1493, 0.2575]])

    bern, likelihood = PPP.detected_update(
        indices,
        z,
        meas_model,
        P_D,
        clutter_intensity,
    )

    np.testing.assert_allclose(likelihood, -4.9618, rtol=0.01)
    np.testing.assert_allclose(bern.state.x,
                               np.array([24.3498, 5.7689, 5.0000, 0, 0.0175]),
                               rtol=0.1)
    ref_P = np.array([
        [0.9779, -0.0122, 0.0000, 0.0000, 0.0000],
        [-0.0122, 0.9707, 0.0000, 0.0000, 0.0000],
        [0.0000, -0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
    ])
    np.testing.assert_almost_equal(
        bern.state.P,
        ref_P,
        decimal=4,
    )


def test_PPP_undetected_update(initial_PPP_intensity_nonlinear):
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_nonlinear)
    P_D = 0.7
    PPP.intensity.weights = np.log(np.array([0.1839, 0.2400, 0.4173]))

    PPP.undetected_update(P_D)

    PPP_weights_ref = np.array([-2.8973, -2.6311, -2.0779])
    np.testing.assert_almost_equal(
        PPP.intensity.weights,
        PPP_weights_ref,
        decimal=4,
    )


def test_PPP_gating(initial_PPP_intensity_linear):
    # Create linear measurement model
    sigma_r = 5.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)

    z = np.array([[0.1, 0.006], [10e6, 10e6]])
    gating_matrix_ud, meas_indices_ud = PPP.gating(z,
                                                   GaussianDensity,
                                                   meas_model,
                                                   gating_size=0.99)
    gating_matrix_ud_ref = np.array([[True, False], [False, False],
                                     [False, False], [False, False]])
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


def test_PPP_get_new_tracks(initial_PPP_intensity_linear):
    # Create linear measurement model
    sigma_r = 5.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)
    P_D = 0.9
    clutter_intensity = 0.7 / 100

    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_linear)
    z = np.array([[0.1, 0.006], [10e6, 10e6]])

    gating_matrix_ud, used_meas_indices_ud = PPP.gating(z,
                                                        GaussianDensity,
                                                        meas_model,
                                                        gating_size=0.99)
    new_tracks = PPP(z, gating_matrix_ud, clutter_intensity, meas_model, P_D)
    assert new_tracks
