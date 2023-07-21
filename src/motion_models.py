import numpy as np

from src.common.gaussian_density import GaussianDensity


class BaseMotionModel:
    def __init__(self, random_state=None, *args, **kwargs):
        self._generator = np.random.RandomState(random_state)

    def f(self, state_vector):
        raise NotImplementedError

    def move(self, params):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class ConstantVelocityMotionModel(BaseMotionModel):
    def __init__(self, dt: float, sigma_q: float, *args, **kwargs):
        """Creates a 2D nearly constant velocity model

        Note:
            the motion model assumes that the state vector x consists of the following states:
            px -> X position
            py -> Y position
            vx -> X velocity
            vy -> Y velocity

        Attributes:
            d (scalar): object state dimension
            F (2 x 2 matrix): function handle return a motion transition function
            Q (4 x 4 matrix): motion noise covariance
            f (4 x 1 vector): function handle return state prediction

        Args:
            T (scalar): sampling time
            sigma (scalar): standart deviation of motion noise
        """

        self.d = 4
        self.dt = dt
        self.sigma = sigma_q
        super(ConstantVelocityMotionModel, self).__init__(*args, **kwargs)

    def f(self, states: np.ndarray, dt: float):
        new_states = np.empty_like(states)
        for i, state in enumerate(states):
            new_states[i] = self._transition_matrix(dt) @ state
        return new_states

    def F(self, state_vector, dt: float):
        return self._transition_matrix(dt=dt)

    def Q(self, dt):
        return self._get_motion_noise_covariance(dt=dt, sigma=self.sigma)

    def move(self, state: GaussianDensity, dt: float = None, if_noisy: bool = False) -> GaussianDensity:
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)
        assert isinstance(state, GaussianDensity), f"Argument of wrong type! Type ={type}"
        if if_noisy:
            next_state = GaussianDensity(
                means=self._generator.multivariate_normal(
                    mean=self._transition_matrix(dt=dt) @ state.means,
                    cov=self._get_motion_noise_covariance(dt=dt),
                ),
                covs=self._get_motion_noise_covariance(),
            )
        else:
            next_state = GaussianDensity(
                means=(self._transition_matrix(dt=dt)[None, ...] @ state.means[..., None])[..., 0],
                covs=self._get_motion_noise_covariance(),
            )
        return next_state

    def _transition_matrix(self, dt=None):
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def _get_motion_noise_covariance(self, dt=None, sigma=None):
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)

        sigma = self.sigma if sigma is None else sigma
        assert isinstance(sigma, float)

        Q = sigma**2 * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4.0 / 4.0, 0.0, dt**3.0 / 2.0],
                [dt**3.0 / 2.0, 0.0, dt**2.0, 0.0],
                [0.0, dt**3.0 / 2.0, 0.0, dt**2.0],
            ]
        )
        return Q

    def __repr__(self) -> str:
        return f"Constant velocity motion model with dt = {self.dt} siqma = {self.sigma} "


class CoordinateTurnMotionModel(BaseMotionModel):
    def __init__(self, dt: float, sigma_V: float, sigma_omega: float, *args, **kwargs):
        """Creates a 2D coordinate turn model with nearly constant polar velocity and turn rate

        Note:
            the motion model assumes that the state vactor x consists of the following states:
            px -> X position
            py -> Y position
            v  -> velocity
            phi -> heading
            omega -> turn rate

        Attributes:
            d (scalar): object state dimension
            F (5 x 5 matrix): function handle return a motion transition function
            Q (5 x 5 matrix): motion noise covariance
            f (5 x 1 vector): function handle return state prediction

        Args:
            T (scalar): sampling time
            sigma_V (scalar): standart deviation of motion noise added to polar velocity
            sigma_omega (scalar): standard deviation of motion noise added to turn rate
        """
        self.d = 5
        self.dt = dt
        self.sigma_V = sigma_V
        self.sigma_omega = sigma_omega

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(d={self.d}, " f"dt={self.dt}, " f"sigma_V={self.sigma_V}, " f"sigma_omega={self.sigma_omega}, ")

    def move(self, state: GaussianDensity, dt: float = None, if_noisy: bool = False) -> GaussianDensity:
        dt = self.dt if dt is None else dt
        assert isinstance(dt, float)
        assert isinstance(state, GaussianDensity), f"Argument of wrong type! Type ={type}"

        if if_noisy:
            next_state = GaussianDensity(
                means=self._generator.multivariate_normal(
                    mean=self._f(state, dt=dt),
                    cov=self.Q(dt=dt),
                ),
                covs=self.get_motion_noise_covariance(),
            )
        else:
            next_state = GaussianDensity(
                means=self._f(state.means, dt=dt),
                covs=self.Q(dt=dt),
            )
        return next_state

    def f(self, state_means, dt):
        next_states = []
        for state_vector in state_means:
            pos_x, pos_y, v, phi, omega = state_vector
            next_state = np.array([dt * v * np.cos(phi), dt * v * np.sin(phi), 0, dt * omega, 0])
            next_states.append(next_state)
        next_means = state_means + np.array(next_states)
        return next_means

    def F(self, state_vectors, dt):
        num_states = state_vectors.shape[0]
        result = np.zeros((state_vectors.shape[0], self.d, self.d))

        for i in range(num_states):
            pos_x, pos_y, v, phi, omega = state_vectors[i]

            result[i] = np.array(
                [
                    [1.0, 0.0, dt * np.cos(phi), -dt * v * np.sin(phi), 0.0],
                    [0.0, 1.0, dt * np.sin(phi), dt * v * np.cos(phi), 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, dt],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )

            # result[i] = np.dot(F_matrix, state_vectors[i])
        return result

    def _f(self, state_means, dt):
        pos_x, pos_y, v, phi, omega = state_means.T

        return np.column_stack(
            [
                pos_x + ((2 * v) / omega) * np.sin(omega * dt / 2) * np.cos(phi + omega * dt / 2),
                pos_y + ((2 * v) / omega) * np.sin(omega * dt / 2) * np.sin(phi + omega * dt / 2),
                v,
                phi + omega * dt,
                omega,
            ]
        )

    def Q(self, dt=None):
        """Motion noise covariance"""
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, self.sigma_V**2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.sigma_omega**2],
            ]
        )  # yapf: disable
