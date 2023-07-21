import numpy as np

from src.common.state import Gaussian
from src.motion_models import ConstantVelocityMotionModel


def test_constant_velocity_motion_model():
    # Create an instance of the ConstantVelocityMotionModel
    dt = 0.1
    sigma_q = 0.5
    random_state = 42
    model = ConstantVelocityMotionModel(random_state, sigma_q)

    # Assert that the instance is created
    assert isinstance(model, ConstantVelocityMotionModel)

    # Create a Gaussian state
    state = Gaussian(x=np.array([1, 2, 3, 4]), P=np.eye(4))

    # Check the f function
    predicted_state = model.f(state.x, dt)
    assert np.allclose(predicted_state, np.array([1.3, 2.4, 3, 4]))

    # Check the F function
    transition_matrix = model.F(state.x, dt)
    expected_transition_matrix = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(transition_matrix, expected_transition_matrix)

    # Check the Q function
    motion_noise_covariance = model.Q(dt)
    expected_motion_noise_covariance = sigma_q**2 * np.array(
        [
            [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
            [0.0, dt**4.0 / 4.0, 0.0, dt**3.0 / 2.0],
            [dt**3.0 / 2.0, 0.0, dt**2.0, 0.0],
            [0.0, dt**3.0 / 2.0, 0.0, dt**2.0],
        ]
    )
    assert np.allclose(motion_noise_covariance, expected_motion_noise_covariance)

    # Check the move function
    moved_state = Gaussian(model.f(state.x, dt), model.Q(dt))
    assert np.allclose(moved_state.x, predicted_state)
    assert np.allclose(moved_state.P, expected_motion_noise_covariance)
