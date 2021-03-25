import numpy as np

from pytest import fixture

from mot.common.gaussian_density import GaussianDensity
from mot.common.state import Gaussian, GaussianMixture, WeightedGaussian
from mot.measurement_models import (
    ConstantVelocityMeasurementModel,
    RangeBearingMeasurementModel,
)
from mot.motion_models import CoordinateTurnMotionModel
from mot.trackers.multiple_object_trackers.PMBM.common import PoissonRFS


@fixture(scope="function")
def initial_PPP_intensity_linear():
    return GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([400.0, -600.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-800.0, -200.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-200.0, 800.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
        ]
    )


@fixture(scope="function")
def initial_PPP_intensity_nonlinear():
    return GaussianMixture(
        [
            WeightedGaussian(
                -0.3861,
                Gaussian(x=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), P=np.eye(5)),
            ),
            WeightedGaussian(
                -0.423,
                Gaussian(x=np.array([20.0, 20.0, -20.0, 0.0, np.pi / 90]), P=np.eye(5)),
            ),
            WeightedGaussian(
                -1.8164,
                Gaussian(
                    x=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), P=np.eye(5)
                ),
            ),
        ]
    )


def test_PPP_predict(initial_PPP_intensity_nonlinear):
    # Create nonlinear motion model (coordinate turn)
    dt = 1.0
    sigma_V = 1.0
    sigma_Omega = np.pi / 180
    motion_model = CoordinateTurnMotionModel(dt, sigma_V, sigma_Omega)

    # Set birth model
    birth_model = GaussianMixture(
        [
            WeightedGaussian(
                w=np.log(0.03),
                gm=Gaussian(
                    x=np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
                    P=np.diag([1.0, 1.0, 1.0, (np.pi / 90) ** 2, (np.pi / 90) ** 2]),
                ),
            )
        ]
        * 4
    )

    # Set probability of esitense
    P_S = 0.8

    # Set Poisson RFS
    PPP = PoissonRFS(initial_intensity=initial_PPP_intensity_nonlinear)

    PPP.predict(motion_model=motion_model, birth_model=birth_model, P_S=P_S)

    PPP_ref_w = np.array(
        [-0.6092, -0.6461, -2.0395, -3.5066, -3.5066, -3.5066, -3.5066]
    )

    PPP_ref_state_x = [
        np.array([5.0000, 0, 5.0000, 0.0175, 0.0175]),
        np.array([0.0, 20.0, -20.0000, 0.0349, 0.0349]),
        np.array([-30.0000, 10.0000, -10.0000, 0.0087, 0.0087]),
        np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
        np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
        np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
        np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
    ]

    PPP_ref_state_P = [
        np.array(
            [
                [2, 0, 1, 0, 0],
                [0, 26, 0, 5, 0],
                [1, 0, 2, 0, 0],
                [0, 5, 0, 2, 1],
                [0, 0, 0, 1, 1.0003],
            ]
        ),
        np.array(
            [
                [2.0000, 0, 1.0000, 0, 0],
                [0, 401.0000, 0, -20.0000, 0],
                [1.0000, 0, 2.0000, 0, 0],
                [0, -20.0000, 0, 2.0000, 1.0000],
                [0, 0, 0, 1.0000, 1.0003],
            ]
        ),
        np.array(
            [
                [2.0000, 0, 1.0000, 0, 0],
                [0, 101.0000, 0, -10.0000, 0],
                [1.0000, 0, 2.0000, 0, 0],
                [0, -10.0000, 0, 2.0000, 1.0000],
                [0, 0, 0, 1.0000, 1.0003],
            ]
        ),
        np.array(
            [
                [1.0000, 0, 0, 0, 0],
                [0, 1.0000, 0, 0, 0],
                [0, 0, 1.0000, 0, 0],
                [0, 0, 0, 0.0012, 0],
                [0, 0, 0, 0, 0.0012],
            ]
        ),
        np.array(
            [
                [1.0000, 0, 0, 0, 0],
                [0, 1.0000, 0, 0, 0],
                [0, 0, 1.0000, 0, 0],
                [0, 0, 0, 0.0012, 0],
                [0, 0, 0, 0, 0.0012],
            ]
        ),
        np.array(
            [
                [1.0000, 0, 0, 0, 0],
                [0, 1.0000, 0, 0, 0],
                [0, 0, 1.0000, 0, 0],
                [0, 0, 0, 0.0012, 0],
                [0, 0, 0, 0, 0.0012],
            ]
        ),
        np.array(
            [
                [1.0000, 0, 0, 0, 0],
                [0, 1.0000, 0, 0, 0],
                [0, 0, 1.0000, 0, 0],
                [0, 0, 0, 0.0012, 0],
                [0, 0, 0, 0, 0.0012],
            ]
        ),
    ]

    np.testing.assert_allclose(PPP.intensity.weights, PPP_ref_w, rtol=0.1)
    np.testing.assert_allclose(
        [gm.x for gm in PPP.intensity.states], PPP_ref_state_x, rtol=0.01
    )
    np.testing.assert_allclose(
        [gm.P for gm in PPP.intensity.states], PPP_ref_state_P, rtol=0.02
    )


def test_PPP_detected_update(initial_PPP_intensity_nonlinear):
    P_D = 0.8
    clutter_intensity = 0.7 / 100

    # Create nonlinear measurement model (range/bearing)
    sigma_r = 5.0
    sigma_b = np.pi / 180
    s = np.array([300, 400]).T
    meas_model = RangeBearingMeasurementModel(
        sigma_r=sigma_r, sigma_b=sigma_b, sensor_pos=s
    )

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
    np.testing.assert_allclose(
        bern.state.x, np.array([24.3498, 5.7689, 5.0000, 0, 0.0175]), rtol=0.1
    )
    ref_P = np.array(
        [
            [0.9779, -0.0122, 0.0000, 0.0000, 0.0000],
            [-0.0122, 0.9707, 0.0000, 0.0000, 0.0000],
            [0.0000, -0.0000, 1.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )
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
    gating_matrix_ud, meas_indices_ud = PPP.gating(
        z, GaussianDensity, meas_model, gating_size=0.99
    )
    gating_matrix_ud_ref = np.array(
        [[True, False], [False, False], [False, False], [False, False]]
    )
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

    gating_matrix_ud, used_meas_indices_ud = PPP.gating(
        z, GaussianDensity, meas_model, gating_size=0.99
    )
    new_tracks = PPP.get_new_tracks(
        z, gating_matrix_ud, clutter_intensity, meas_model, P_D
    )
    assert new_tracks
