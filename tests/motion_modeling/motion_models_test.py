import unittest

import numpy as np

from src.common.gaussian_density import GaussianDensity as GaussianDensity
from src.motion_models import ConstantVelocityMotionModel


class Test_BaseMotionModels(unittest.TestCase):
    def test_constant_velocity_motion_model(self):
        delta_t = 1.0
        sigma_q = 2.0
        CV_motion_model = ConstantVelocityMotionModel(dt=delta_t, sigma_q=sigma_q)

        initial_state = np.array([0.0, 0.0, 5.0, 5.0])
        expected_predicted_array = np.array([[5.0, 5.0, 5.0, 5.0]])
        predicted_state = CV_motion_model.move(state=GaussianDensity(means=initial_state, covs=np.eye(4)))

        np.testing.assert_array_equal(
            predicted_state.means,
            expected_predicted_array,
            err_msg=f"Wrong prediction state! {predicted_state}",
        )
