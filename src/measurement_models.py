from typing import Annotated as Ann

import numpy as np


class MeasurementModel:
    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)
        self.dim: float = None  # Dimension of the measurement vector (meas_dim)
        self.R: Ann[np.ndarray, "meas_dim, meas_dim"] = None  # Measurement noise covariance matrix

    def H(self, states) -> Ann[np.ndarray, "n, meas_dim, states_dim"]:
        """Handle to generate observation matrix"""
        raise NotImplementedError

    def h(self, states: Ann[np.ndarray, "n, states_dim"]) -> Ann[np.ndarray, "n, meas_dim"]:
        return np.einsum("ijk,ik->ij", self.H(states), states)  # batched self.H(state_vector) @ state_vector

    def observe(self, states) -> Ann[np.ndarray, "n, meas_dim"]:
        """Add noise to measurements"""
        return self.h(states) + self._generator.multivariate_normal(mean=np.zeros(self.dim), cov=self.R, size=states.shape[0])


class CoordinateTurnMeasurementModel(MeasurementModel):
    def __init__(self, sigma, *args, **kwargs):
        """Creates the measurement model for a 2D coordinate turn motion model

        Args:
            sigma (scalar): standart deviation of measurement noise

        Attributes:
            d (scalar): measurement dimenstion
            H (2 x 5 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement

        Notes: the first two entries of the state vector represents
               the X-position and Y-position, respectively.

        Returns:
            self (MeasurementModel)
        """

        self.dim = 2
        self.R = (sigma**2) * np.eye(2)

        def H(self, states):
            return np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])


class ConstantVelocityMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r, *args, **kwargs):
        """Creates the measurement model for a 2D nearly constant velocity motion model

        Args:
            sigma (scalar): standart deviation of a measurement noise

        Attributes:
            d (scalar): measurement dimenstion
            H (2 x 4 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement

        Returns:
            self (MeasurementModel)
        """
        super().__init__(*args, **kwargs)
        self.dim = 2
        self.R = (sigma_r**2) * np.eye(2)

    def H(self, states: Ann[np.ndarray, "n, 4"]) -> Ann[np.ndarray, "n, 2, 4"]:
        return np.repeat(
            np.expand_dims(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]), axis=0),
            states.shape[0],
            axis=0,
        )


class NuscenesConstantVelocityMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r, *args, **kwargs):
        """Creates the measurement model for a 2D nearly constant velocity motion model

        Args:
            sigma (scalar): standart deviation of a measurement noise

        Attributes:
            d (scalar): measurement dimenstion
            H (2 x 4 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement

        Returns:
            self (MeasurementModel)
        """
        super().__init__(*args, **kwargs)
        self.dim = 3
        self.R = (sigma_r**2) * np.eye(3)

    def H(self, states):
        return np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)


class RangeBearingMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r: float, sigma_b: float, sensor_pos, *args, **kwargs):
        """Creats the range/bearnig measurement model
        sigma_r : scalar
            standart deviation of measurement noise added to range
        sigma_b : scalar
            standart deviation of measurement noise added to bearing
        sensor_pos : np.array (2 x 1)
            sensor positions
        """
        super().__init__(*args, **kwargs)

        self.dim = 2
        self.sigma_r = sigma_r
        self.sigma_b = sigma_b
        self.R = np.diag([np.power(self.sigma_r, 2), np.power(self.sigma_b, 2)])
        self.sensor_pos = sensor_pos

    def H(self, states):
        """
        [(state_pos_x - sensor_pose_x) / range, (state_pos_y - sensor_pose_y) / rng, 0.0, 0.0, 0.0],
        [-(state_pos_y - sensor_pose_y) / rng**2, (state_pos_x - sensor_pose_x / rng**2, 0.0, 0.0, 0.0,
        """
        H = np.zeros((states.shape[0], self.dim, 5))
        H[:, 0, 0] = (states[..., 0] - self.sensor_pos[0]) / self._get_range(states)
        H[:, 0, 1] = (states[..., 1] - self.sensor_pos[1]) / self._get_range(states)
        H[:, 1, 0] = (-states[..., 1] - self.sensor_pos[1]) / np.power(self._get_range(states), 2)
        H[:, 1, 1] = (states[..., 0] - self.sensor_pos[0]) / np.power(self._get_range(states), 2)
        return H

    def _get_range(self, states):
        return np.linalg.norm(states[..., :2] - self.sensor_pos, axis=-1)

    def _get_bearing(self, states):
        return np.arctan2(states[..., 1] - self.sensor_pos[1], states[..., 0] - self.sensor_pos[0])
