from abc import ABC

import numpy as np


class MeasurementModel(ABC):
    """
    MeasurementModel is a abstract class for different measurement models.
    """

    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)

    def observe(self, params):
        pass

    @property
    def dim(self):
        pass


import numpy as np


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

        self.d = 2
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
        self.R = (sigma**2) * np.eye(2)

    def __call__(self, x):
        return self.H @ x

    @property
    def dim(self):
        return self.d


from nptyping import NDArray, Shape, Float
import numpy as np


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

        self.d = 2
        self.R = (sigma_r**2) * np.eye(2)
        super(ConstantVelocityMeasurementModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.H @ x

    def _observation_matrix(self, state_vector):
        return self.H(state_vector)

    @property
    def _noise_covariance(self):
        return self.R

    def observe(self, state_vector: NDArray[Shape["N, 4"], Float]) -> NDArray[Shape["N, 4"], Float]:
        means: NDArray[Shape["N, 2"], Float] = np.einsum("ijk,ik->ij", self._observation_matrix(state_vector), state_vector)
        samples = np.array([self._generator.multivariate_normal(means[i], self._noise_covariance) for i in range(means.shape[0])])
        return samples

    def H(self, vectors: NDArray[Shape["N, 4"], Float]) -> NDArray[Shape["N, 2, 4"], Float]:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        return np.repeat(np.expand_dims(np.array(H), axis=0), vectors.shape[0], axis=0)

    def h(self, state_vector: NDArray[Shape["N, 4"], Float]) -> NDArray[Shape["N, 2"], Float]:
        return np.einsum("ijk,ik->ij", self.H(state_vector), state_vector)  # batched self.H(state_vector) @ state_vector

    @property
    def dim(self):
        return self.d


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

        self.d = 3
        self.R = (sigma_r**2) * np.eye(3)
        super(ConstantVelocityMeasurementModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.H @ x

    def _observation_matrix(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        return self.H(state_vector)

    @property
    def _noise_covariance(self):
        return self.R

    def observe(self, state_vector):
        assert isinstance(state_vector, np.ndarray)
        Z = self._generator.multivariate_normal(
            mean=self._observation_matrix(state_vector) @ state_vector,
            cov=self._noise_covariance,
        )
        return Z

    def H(self, state_vector):
        return np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)

    def h(self, state_vector):
        return self.H(state_vector) @ state_vector

    @property
    def dim(self):
        return self.d


import numpy as np


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

        self.d = 2
        self.sigma_r = sigma_r
        self.sigma_b = sigma_b
        self.R = np.diag([np.power(self.sigma_r, 2), np.power(self.sigma_b, 2)])
        self.sensor_pos = sensor_pos
        super(RangeBearingMeasurementModel, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.H @ x

    @property
    def _observation_matrix(self):
        return self.H()

    @property
    def _noise_covariance(self):
        return self.R

    def observe(self, state_vector):
        observation = np.array([self._get_range(state_vector), self._get_bearing(state_vector)]).T
        return observation

    def _get_range(self, state_vector):
        return np.linalg.norm(state_vector[..., :2] - self.sensor_pos, axis=-1)

    def _get_bearing(self, state_vector):
        return np.arctan2(state_vector[:, 1] - self.sensor_pos[1], state_vector[:, 0] - self.sensor_pos[0])

    def H(self, state_vector):
        # yapf: disable
        x, s, rng = state_vector, self.sensor_pos, self._get_range(state_vector)
        H = np.array([
            [(x[0] - s[0]) / rng, (x[1] - s[1]) / rng, 0.0, 0.0, 0.0],
            [-(x[1] - s[1]) / rng**2, (x[0] - s[0]) / rng**2, 0.0, 0.0, 0.0],
        ])
        # yapf: enable
        return H

    @property
    def dim(self):
        return self.d

    def h(self, state_vector: np.ndarray) -> np.ndarray:
        """Handle to generate measurement

        Parameters
        ----------
        state_vector : np.ndarray
            state vector

        Returns
        -------
        np.ndarray
            state_vector
        """
        return np.array([self._get_range(state_vector), self._get_bearing(state_vector)])
