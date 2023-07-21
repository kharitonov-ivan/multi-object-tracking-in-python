import numpy as np


class MotionModel:
    def __init__(self, random_state: int, d: int, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)
        self.d = d

    def f(self, state_vector: np.ndarray, dt: float) -> np.ndarray:
        """Calculates next state vector from current"""
        return self.F(state_vector, dt) @ state_vector

    def F(self, state_vector: np.ndarray, dt: float) -> np.ndarray:
        """Represents transition matrix"""
        raise NotImplementedError

    def Q(self, dt: float) -> np.ndarray:
        """Represents motion noise covariance"""
        raise NotImplementedError


class ConstantVelocityMotionModel(MotionModel):
    def __init__(self, random_state: int, sigma_q: float, *args, **kwargs):
        super().__init__(random_state, d=4)  # 4 states: x, y, vx, vy
        self.sigma = sigma_q

    def F(self, state_vector: np.ndarray, dt: float) -> np.ndarray:
        """Transition matrix for constant velocity model"""
        return np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def Q(self, dt: float) -> np.ndarray:
        """Motion noise covariance for constant velocity model"""
        return self.sigma**2 * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4.0 / 4.0, 0.0, dt**3.0 / 2.0],
                [dt**3.0 / 2.0, 0.0, dt**2.0, 0.0],
                [0.0, dt**3.0 / 2.0, 0.0, dt**2.0],
            ]
        )


class CoordinateTurnMotionModel(MotionModel):
    def __init__(self, random_state: int, sigma_v: float, sigma_omega: float, *args, **kwargs):
        super().__init__(random_state, d=5)  # 5 states: x, y, v, phi, omega
        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega

    def F(self, state_vector: np.ndarray, dt: float) -> np.ndarray:
        """Transition matrix for coordinate turn model"""
        pos_x, pos_y, v, phi, omega = state_vector
        return np.array(
            [
                [1.0, 0.0, dt * np.cos(phi), -dt * v * np.sin(phi), 0],
                [0.0, 1.0, dt * np.sin(phi), dt * v * np.cos(phi), 0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

    def Q(self, dt: float) -> np.ndarray:
        """Motion noise covariance for coordinate turn model"""
        return np.diag([0, 0, self.sigma_v**2, 0, self.sigma_omega**2])


class ConstantAccelerationMotionModel(MotionModel):
    def __init__(self, random_state: int, sigma_a: float, *args, **kwargs):
        super().__init__(random_state, d=6)  # 6 states: x, y, vx, vy, ax, ay
        self.sigma_a = sigma_a

    def F(self, state_vector: np.ndarray, dt: float) -> np.ndarray:
        """Transition matrix for constant acceleration model"""
        return np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt**2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt**2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

    def Q(self, dt: float) -> np.ndarray:
        """Motion noise covariance for constant acceleration model"""
        sigma_a2 = self.sigma_a**2
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        return (
            np.array(
                [
                    [dt4 / 4, 0, dt3 / 2, 0, dt2 / 2, 0],
                    [0, dt4 / 4, 0, dt3 / 2, 0, dt2 / 2],
                    [dt3 / 2, 0, dt2, 0, dt, 0],
                    [0, dt3 / 2, 0, dt2, 0, dt],
                    [dt2 / 2, 0, dt, 0, 1, 0],
                    [0, dt2 / 2, 0, dt, 0, 1],
                ]
            )
            * sigma_a2
        )
