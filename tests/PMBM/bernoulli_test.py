import numpy as np

from mot.common.state import Gaussian
from mot.measurement_models import RangeBearingMeasurementModel
from mot.motion_models import CoordinateTurnMotionModel
from mot.trackers.multiple_object_trackers.PMBM.common.bernoulli import Bernoulli


def test_bern_predict():
    # Create nonlinear motion model (coordinate turn)
    dt = 1.0
    sigma_V = 1.0
    sigma_Omega = np.pi / 180
    motion_model = CoordinateTurnMotionModel(dt, sigma_V, sigma_Omega)

    # Set probability of existence
    survival_probability = 0.8

    # Set Bernoulli RFS
    bern = Bernoulli(
        r=0.9058,
        initial_state=Gaussian(
            x=np.array([0.1270, 0.9134, 0.6324, 0.0975, 0.2785]), P=np.eye(5)
        ),
    )

    got_bern = Bernoulli.predict(bern=bern, motion_model=motion_model, survival_probability=survival_probability)
    # reference
    r_ref = 0.7380
    state_ref = Gaussian(
        x=np.array([0.7563, 0.9750, 0.6324, 0.3760, 0.2785]),
        P=np.array(
            [
                [1.9943, 0.0582, 0.9952, -0.0616, 0],
                [0.0582, 1.4056, 0.0974, 0.6294, 0],
                [0.9952, 0.0974, 2.0000, 0, 0],
                [-0.0616, 0.6294, 0, 2.0, 1.0],
                [0, 0, 0, 1.0000, 1.0003],
            ]
        ),
    )
    np.testing.assert_allclose(r_ref, got_bern.r, rtol=0.05)
    np.testing.assert_allclose(state_ref.x, got_bern.state.x, atol=1e-4)
    np.testing.assert_allclose(state_ref.P, got_bern.state.P, atol=1e-4)


def test_bern_undetected_update():
    # set detection probability
    P_D = 0.8

    # create hypotheses tree
    bern = bern = Bernoulli(
        r=0.6,
        initial_state=Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0, 0.0]), P=np.eye(5)),
    )
    ref_bern_r = 0.2308
    ref_lik_undetected = -0.6539

    new_bern, likelihood_undetected = Bernoulli.undetected_update(
        bern,
        P_D=P_D,
    )

    np.testing.assert_allclose(new_bern.r, ref_bern_r, rtol=1e-3)
    np.testing.assert_allclose(likelihood_undetected, ref_lik_undetected, rtol=1e-3)


def test_bern_detected_update_likelihood():
    # set detection probability
    P_D = 0.8

    # create hypotheses tree
    bern = Bernoulli(
        r=0.6,
        initial_state=Gaussian(
            x=np.array([0.1270, 0.9134, 0.6324, 0.0975, 0.2785]), P=np.eye(5)
        ),
    )

    z = np.array([[0.5469, 0.9649, 0.9706], [0.9575, 0.1576, 0.9572]]).T

    # create nonlinear measurement model (range/bearing)
    sigma_r = 5.0
    sigma_b = np.pi / 180
    s = np.array([300, 400])
    meas_model = RangeBearingMeasurementModel(
        sigma_r=sigma_r, sigma_b=sigma_b, sensor_pos=s
    )

    likelihood_detected = Bernoulli.detected_update_likelihood(bern, z, meas_model, P_D)

    likelihood_detected_ref = 1e3 * np.array([-1.4883, -0.0409, -1.4872])
    np.testing.assert_allclose(likelihood_detected, likelihood_detected_ref, rtol=0.05)


def test_bern_update_state():
    # create hypotheses tree
    bern = Bernoulli(
        r=0.6,
        initial_state=Gaussian(
            x=np.array([0.0357, 0.8491, 0.9340, 0.6787, 0.7577]), P=np.eye(5)
        ),
    )

    z = np.array([0.6948, 0.3171]).reshape([1, 2])

    # create nonlinear measurement model (range/bearing)
    sigma_r = 5.0
    sigma_b = np.pi / 180
    s = np.array([300, 400]).T
    meas_model = RangeBearingMeasurementModel(
        sigma_r=sigma_r, sigma_b=sigma_b, sensor_pos=s
    )

    tested_bern = Bernoulli.detected_update_state(
        bern,
        z=z,
        meas_model=meas_model,
    )
    ref_r = 1.0
    ref_state_x = np.array([24.6940, 6.3070, 0.9340, 0.6787, 0.7577])
    ref_state_P = np.array(
        [
            [0.9778, -0.0122, 0.0, 0.0, 0.0],
            [-0.0122, 0.9707, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0000, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0000, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0000],
        ]
    )

    np.testing.assert_allclose(tested_bern.r, ref_r, rtol=0.01)
    np.testing.assert_allclose(tested_bern.state.x, ref_state_x, rtol=0.01)
    np.testing.assert_allclose(tested_bern.state.P, ref_state_P, rtol=0.01)
