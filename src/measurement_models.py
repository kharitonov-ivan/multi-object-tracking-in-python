import numpy as np


class MeasurementModel:
    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)

    def __repr__(self):
        raise NotImplementedError

    def R(self) -> np.ndarray:
        """Measurement noise covariance matrix"""
        raise NotImplementedError

    def H(self, state_vector: np.ndarray) -> np.ndarray:
        """Observation matrix"""
        raise NotImplementedError

    def h(self, state_vector: np.ndarray) -> np.ndarray:
        """Function generate measurement from state"""
        raise NotImplementedError

    def observe(self, state_vector):
        return self._generator.multivariate_normal(
            mean=((self.H(state_vector) @ state_vector.T).T),
            cov=self.R,
        )


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
        self.dim = 2
        self.R = (sigma_r**2) * np.eye(2)
        super().__init__(*args, **kwargs)

    def H(self, state_vector):
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def h(self, state_vector):
        return self.H(state_vector) @ state_vector


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

        self.dim = 3
        self.R = (sigma_r**2) * np.eye(3)
        super().__init__(*args, **kwargs)

    def H(self, state_vector):
        return np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)

    def h(self, state_vector):
        return self.H(state_vector) @ state_vector


class RangeBearingMeasurementModel(MeasurementModel):
    def __init__(self, sigma_r: float, sigma_b: float, sensor_pos, *args, **kwargs):
        """Creats the range/bearnig measurement model

        Parameters
        ----------
        sigma_r : scalar
            standart deviation of measurement noise added to range
        sigma_b : scalar
            standart deviation of measurement noise added to bearing
        sensor_pos : np.array (2 x 1)
            sensor positions

        Attributes
        -------
        sigma_r : scalar
            standart deviation of measurement noise added to range
        # TODO beatify
        d (scalar): measurement dimenstion
            H (2 x 4 matrix): function handle return an observation matrix
            R (2 x 2 matrix): measurement noise covatiance
            h (2 x 1 matrix): function handle return a measurement


        Returns
        -------
        [type]
            [description]
        """

        self.dim = 2
        self.sigma_r = sigma_r
        self.sigma_b = sigma_b
        self.R = np.diag([np.power(self.sigma_r, 2), np.power(self.sigma_b, 2)])
        self.sensor_pos = sensor_pos
        super().__init__(*args, **kwargs)

    def H(self, state_vector: np.ndarray) -> np.ndarray:
        x, s, rng = state_vector, self.sensor_pos, self._get_range(state_vector)
        return np.array(
            [
                [(x[0] - s[0]) / rng, (x[1] - s[1]) / rng, 0.0, 0.0, 0.0],
                [-(x[1] - s[1]) / rng**2, (x[0] - s[0]) / rng**2, 0.0, 0.0, 0.0],
            ]
        )

    def h(self, state_vector: np.ndarray) -> np.ndarray:
        return np.array([self._get_range(state_vector), self._get_bearing(state_vector)])

    def _get_range(self, state_vector: np.ndarray):
        assert isinstance(state_vector, np.ndarray)
        return np.linalg.norm(state_vector[:2] - self.sensor_pos)

    def _get_bearing(self, state_vector: np.ndarray):
        return np.arctan2(state_vector[..., 1] - self.sensor_pos[1], state_vector[..., 0] - self.sensor_pos[0])
