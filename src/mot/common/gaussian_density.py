from __future__ import annotations
import copy
import logging
import typing as tp
from nptyping import NDArray, Shape, Float


import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
from sklearn.covariance import log_likelihood  # noqa: I201
import mot
from mot.common.normalize_log_weights import normalize_log_weights
from mot.measurement_models import MeasurementModel
from mot.utils.vectorized_gaussian_logpdf import vectorized_gaussian_logpdf
import numba as nb


def make_SPD(covariance: np.ndarray) -> np.ndarray:
    return 0.5 * (covariance + covariance.swapaxes(-1, -2))


import numpy as np
import torch


def invert_pd_matrix_torch(A):
    # Разложение Холецкого
    L = torch.cholesky(A)
    # Используем cholesky_inverse для получения обратной матрицы
    A_inv = torch.cholesky_inverse(L)

    return A_inv


def invert_pd_matrix_np_torch(A_np):
    # Преобразовываем numpy array в torch tensor
    A_torch = torch.from_numpy(A_np)

    # Вычисляем обратные матрицы с помощью torch
    A_inv_torch = invert_pd_matrix_torch(A_torch)

    # Конвертируем результат обратно в numpy array
    A_inv_np = A_inv_torch.numpy()

    return A_inv_np


from scipy.linalg import cholesky


def invert_pd_matrix(A):
    # Выполним разложение Холецкого
    L = cholesky(A, lower=True)
    # Найдем обратную матрицу для L
    L_inv = np.linalg.inv(L)
    # Обратная матрица A будет произведением L_inv.T и L_inv
    A_inv = L_inv.T @ L_inv

    return A_inv


class GaussianDensity:
    def __init__(
        self,
        means: NDArray[Shape["N_components, State_dim"], Float] = None,
        covs: NDArray[Shape["N_components, State_dim, State_dim"], Float] = None,
        weights: tp.Optional[NDArray[Shape["N_components"], Float]] = None,
    ):
        if means is not None and covs is not None:
            assert isinstance(means, np.ndarray), "Argument means must be np.ndarray"
            assert isinstance(covs, np.ndarray), "Argument covs must be np.ndarray"
            if means.ndim == 1:
                means = means[None, ...]
            assert means.ndim == 2, "Means must be 2D array"
            if covs.ndim == 2:
                covs = covs[None, ...]
            assert covs.ndim == 3, "Covariances must be 3D array"
            if weights is not None and weights.ndim == 0:
                weights = np.array([weights])
            assert means.shape[0] == covs.shape[0], "Number of components must be equal"
            if weights is not None:
                assert means.shape[0] == weights.shape[0], "Number of components must be equal"
            assert means.shape[-1] == covs.shape[-1], "State dim in means and covs must be equal"
            assert covs.shape[-1] == covs.shape[-2], "Covariance matrix should be square!"
        self.means = means.copy() if means is not None else None
        self.covs = covs.copy() if covs is not None else None
        self.weights = weights.copy() if weights is not None else None

    def is_empty(self) -> bool:
        """Проверяет, является ли объект GaussianDensity пустым."""
        return self.means is None and self.covs is None and self.weights is None

    def __add__(self, other):
        if self.is_empty():
            return other
        elif other.is_empty():
            return self
        else:
            means = np.concatenate((self.means, other.means), axis=0)
            covs = np.concatenate((self.covs, other.covs), axis=0)
            weights = (
                np.concatenate((self.weights, other.weights), axis=0) if self.weights is not None and other.weights is not None else None
            )
            assert means.shape[0] == covs.shape[0]
            if weights is not None:
                assert means.shape[0] == weights.shape[0]
            return GaussianDensity(means, covs, weights)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return GaussianDensity(
                means=self.means[idx],
                covs=self.covs[idx],
                weights=self.weights[idx] if self.weights is not None else None,
            )
        else:
            return GaussianDensity(
                means=self.means[idx][None, ...],
                covs=self.covs[idx][None, ...],
                weights=self.weights[idx][None, ...] if self.weights is not None else None,
            )

    def __setitem__(self, idx, value):
        if self.is_empty():
            raise ValueError("Cannot modify an empty GaussianDensity object.")

        if not isinstance(value, GaussianDensity):
            raise TypeError("Value must be an instance of GaussianDensity.")

        self.means[idx] = value.means[0]
        self.covs[idx] = value.covs[0]
        if self.weights is not None and value.weights is not None:
            self.weights[idx] = value.weights

    @property
    def size(self):
        return self.means.shape[0] if self.means is not None else 0

    def __len__(self):
        return self.size

    @staticmethod
    def predict(gaussians: GaussianDensity, motion_model: BaseBaseMotionModel, dt: float) -> Gaussian:
        """Performs linear/nonlinear (Extended) Kalman prediction step

        Args:
            state (Gaussian): a structure with two fields - mean and covariance
            motion_model (BaseBaseMotionModel): a structure specifies the motion model parameters

        Returns:
            state_pred (Gaussian): predicted value of state
        """
        next_x = motion_model.f(gaussians.means, dt)
        next_F = motion_model.F(gaussians.means, dt)
        next_P = next_F @ gaussians.covs @ next_F.T + motion_model.Q(dt)
        return GaussianDensity(next_x, next_P, gaussians.weights)

    # @staticmethod
    # def update_states_with_likelihoods_by_single_measurement(
    #     initial_states: GaussianMixture,
    #     measurement: np.ndarray,
    #     measurement_model: MeasurementModel,
    # ):

    #     H_x = measurement_model.H(initial_states.means)
    #     # Innovation covariance
    #     S = H_x @ initial_states.covs @ H_x.T + measurement_model.R

    #     # Make sure matrix S is positive definite
    #     S = 0.5 * (S + np.transpose(S, axes=(0, 2, 1)))

    #     K = initial_states.covs @ H_x.T @ np.linalg.inv(S)

    #     measurement_row = np.vstack([measurement] * initial_states.size)
    #     fraction = measurement_row - measurement_model.h(initial_states.means.T).T
    #     with_K = np.einsum("ijk,ik->ij", K, fraction)
    #     new_states = initial_states.means + with_K

    #     state_vector_size = initial_states.means[0].shape[-1]

    #     next_covariances = (np.eye(state_vector_size) - K @ H_x) @ initial_states.covs

    #     next_states = [Gaussian(new_states[idx], next_covariances[idx]) for idx in range(initial_states.size)]

    #     measurements_bar = np.expand_dims(H_x, axis=0) @ initial_states.means.T

    #     # it takes 0.0277 sec
    #     # TODO: clean up
    #     # loglikelihoods = [
    #     #     multivariate_normal.logpdf(measurement, measurements_bar[0].T[idx], S[idx])
    #     #     for idx in range(initial_states.size)
    #     # ]

    #     loglikelihoods_fast = vectorized_gaussian_logpdf(
    #         data_points=measurement_row,
    #         means=measurements_bar.squeeze().T,
    #         covariances=np.diagonal(S, axis1=2),
    #     )

    #     # np.testing.assert_almost_equal(loglikelihoods, loglikelihoods_fast)

    #     return next_states, loglikelihoods_fast

    @staticmethod
    def get_Kalman_gain(
        initial_states: GaussianMixture, measurement_model: MeasurementModel
    ) -> Tuple[NDArray[Shape["N, 2, State_dim"], Float], NDArray[Shape["N, 2, 2"], Float], NDArray[Shape["N, State_dim, 2"], Float],]:
        H_x = measurement_model.H(initial_states.means)  # Measurement model Jacobian
        H_x_T = np.swapaxes(H_x, -1, -2)
        S = H_x @ initial_states.covs @ H_x_T + measurement_model.R  # Innovation covariance
        S = make_SPD(S)  # Make sure matrix S is positive definite

        S_inv = np.linalg.inv(S)  # invert_pd_matrix_np_torch(S)
        K = (initial_states.covs @ H_x_T) @ S_inv  # Kalman gain

        # Predict next measurement using current state
        z_bar = (H_x @ initial_states.means[..., None])[..., 0]
        return H_x, S, K, z_bar

    @staticmethod
    def update(
        initial_state: GaussianMixture,
        measurements: NDArray[Shape["N, Measurement_dim"], Float],
        model_measurement: MeasurementModel,
        H_x: tp.Optional[NDArray[Shape["N, Measurement_dim, State_dim"], Float]] = None,
    ) -> Tuple[
        NDArray[Shape["N_gaussians, N_measurements, State_dim"], Float],
        NDArray[Shape["N_gaussians, N_measurements, State_dim, State_dim"], Float],
    ]:
        """
        Performs linear/nonlinear (Extended) Kalman update step
        """

        if not H_x:
            H_x, S, K, z_bar = GaussianDensity.get_Kalman_gain(initial_state, model_measurement)

        measurement_by_states: NDArray[Shape["N_gaussians, N_measurements, Dim_measurement"], Float] = np.repeat(
            np.expand_dims(measurements, 0), initial_state.size, axis=0
        )

        fraction: NDArray[Shape["N_gaussians, N_measurements, Dim_measurement"], Float] = (
            measurement_by_states - model_measurement.h(initial_state.means)[:, None, :]
        )

        next_means = initial_state.means[:, None, ...] + np.einsum("ijkl, ijl -> ijk", K[:, None, ...], fraction)
        state_vector_size = initial_state.means.shape[-1]
        next_covariances = (np.eye(state_vector_size) - K @ H_x) @ initial_state.covs
        next_covariances = np.repeat(next_covariances[:, None, ...], len(measurements), axis=1)  # TODO: check if it is correct

        return next_means, next_covariances, (H_x, S, K, z_bar)

    # @staticmethod
    # def update_likelihoods_vectorized(
    #     updated_states,
    #     updated_covariances,
    #     measurements: np.ndarray,
    #     measurement_model: MeasurementModel,
    # ):
    #     H_x = measurement_model.H(updated_states)
    #     # Innovation covariance

    #     S = H_x @ updated_covariances @ H_x.T + measurement_model.R

    #     # Make sure matrix S is positive definite
    #     S = 0.5 * (S + np.transpose(S))

    #     # it takes 0.0277 sec
    #     # # TODO: clean up
    #     bar = H_x @ updated_states.T

    #     # loglikelihoods = [
    #     #     scipy.stats.multivariate_normal.logpdf(measurements[idx], bar.T[idx], S)
    #     #     for idx in range(len(measurements))
    #     # ]

    #     loglikelihoods = vectorized_gaussian_logpdf(
    #         data_points=measurements,
    #         means=bar.T,
    #         covariances=np.vstack([np.diagonal(S)] * len(measurements)),
    #     )
    #     return loglikelihoods

    @staticmethod
    def predict_loglikelihood(
        state_pred: Gaussian,
        measurements: NDArray[Shape["N_measurements, Measurement_dim"], Float],
        measurement_model: MeasurementModel,
        H_X_S=None,
    ) -> NDArray[Shape["N_gaussians, N_measurements"], Float]:
        """Calculates the predicted likelihood in logarithm domain
        Returns:
            predicted_loglikelihood (np.ndarray (number of measurements)): predicted likelihood
                                                                        for each measurement
        """
        # Jacobian
        (H_x, S, K, z_bar) = H_X_S if H_X_S else GaussianDensity.get_Kalman_gain(state_pred, measurement_model)

        # Difference between measurement and predicted measurement using inovation of current state
        log_likelihood = vectorized_gaussian_logpdf(measurements, z_bar, S)
        assert log_likelihood.shape[1] == measurements.shape[0]
        return log_likelihood

    @staticmethod
    def ellipsoidal_gating(
        gaussians: GaussianDensity,
        measurements: npt.ArrayLike,
        measurement_model: MeasurementModel,
        confidence_level: float,
        H_x=None,
        S_inv=None,
    ) -> np.ndarray:
        """Performs ellipsoidal gating for multiple objects.

        Args:
            means (np.ndarray (N_gaussians, State_dim)): means of the Gaussian components
            covs (np.ndarray (N_gaussians, State_dim, State_dim)): covariances of the Gaussian components
            measurements (npt.ArrayLike (N_measurements, Measurement_dim)): measurements to be gated
            measurement_model (MeasurementModel): specifies the measurement model parameters
            confidence_level (float): confidence level for the gating
            H_x (Optional[np.ndarray (N_gaussians, N_measurements, Measurement_dim, State_dim)]): measurement model Jacobian
            S_inv (Optional[np.ndarray (N_gaussians, Measurement_dim, Measurement_dim)]): inverse of the innovation covariance

        Returns:
            np.ndarray (N_gaussians, N_measurements): boolean mask indicating whether each measurement is inside the gate
            np.ndarray (N_gaussians, N_measurements): Mahalanobis distance of each measurement to the predicted measurement
        """
        if len(measurements) == 0:
            return np.array([]), None

        assert (
            measurements.shape[-1] == measurement_model.dim
        ), "Measurement dimension must be equal to the dimension of the measurement model"

        assert 0.0 < confidence_level < 1.0, "Confidence level must be in (0, 1)"
        gating_size = chi2.ppf(confidence_level, df=measurement_model.d)

        if H_x is None or S_inv is None:
            H_x = measurement_model.H(gaussians.means)  # Measurement model Jacobian
            H_x_T = np.swapaxes(H_x, -1, -2)
            S = H_x @ gaussians.covs @ H_x_T + measurement_model.R  # Innovation covariance
            S = make_SPD(S)  # Make sure matrix S is positive definite
            S_inv = np.linalg.inv(S)

        # Predict next measurement using current state
        z_bar = (H_x @ gaussians.means[..., None])[..., 0]

        # Difference between measurement and prediction
        z_diff = np.repeat(np.expand_dims(measurements, 0), z_bar.shape[0], axis=0) - z_bar[:, None, :]

        n_gaussians, n_measurements = len(gaussians), len(measurements)

        mahalanobis_dist = np.empty((n_gaussians, n_measurements))
        for i in range(n_gaussians):
            mahalanobis_dist[i, :] = [z_diff[i, j, :] @ S_inv[i] @ z_diff[i, j, :].T for j in range(n_measurements)]

        mask = mahalanobis_dist < gating_size
        return mask, mahalanobis_dist

    # @staticmethod
    # def moment_matching(weights: List[float], states: List[Gaussian]) -> Gaussian:
    #     """Aproximates a Gaussian mixture density as a single Gaussian using moment matching

    #     Args:
    #         weights (List(float)): normalized weight of Gaussian components
    #                                in logarithm domain
    #         states (List(Gaussian)): list of Gaussians

    #     Returns:
    #         Gaussian: resulted mixture
    #     """
    #     if len(weights) == 0:
    #         return

    #     log_weights = np.exp(weights)

    #     # Mean of the states
    #     N = len(weights)
    #     x_bar = np.zeros(states[0].means.shape[0])

    #     for state, weight in zip(states, log_weights):
    #         x_bar += weight * state.means

    #     # Covariance
    #     P_bar = np.zeros_like(states[0].covs)
    #     for idx in range(N):
    #         d = x_bar - states[idx].means
    #         P_bar += (states[idx].covs + d @ d.T) * log_weights[idx]

    #     matched_state = Gaussian(means=x_bar, covs=P_bar.T)
    #     return matched_state

    @staticmethod
    def moment_matching(log_weights: List[float], states: List[Gaussian]) -> Gaussian:
        """Aproximates a Gaussian mixture density as a single Gaussian using moment matching

        Args:
            weights (List(float)): normalized weight of Gaussian components
                                   in logarithm domain
            states (List(Gaussian)): list of Gaussians

        Returns:
            Gaussian: resulted mixture
        """
        weights = np.exp(log_weights)
        mean_mixture = np.average(states.means, axis=0, weights=weights)  # aka x_bar
        P_avg_cov = np.average(states.covs, axis=0, weights=weights)  # aka P_bar
        delta = mean_mixture - states.means
        ps = np.einsum("ij, ji -> i", delta, delta.T)
        mean_spread = np.average(ps, axis=0, weights=weights)  # weight.*((mean_mixture - mean)*(mean_mixture - mean)');
        return GaussianDensity(means=mean_mixture, covs=(P_avg_cov + mean_spread))

    @staticmethod
    def mixture_reduction(weights, states, threshold):
        """Uses a greedy merging method to reduce the number of Gaussian components
        for a Gaussian mixture density

        Args:
            weights ([type]): normalized weight of Gaussian components in logarithmic scale
            states ([type]): [description]
            threshold ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_log_weights = []
        new_merged_states = []

        states = copy.deepcopy(states)
        weights = copy.deepcopy(weights)
        while states:
            # Find the component with the highest weight
            max_weight_idx = np.argmax(weights)
            try:
                S = np.linalg.inv(states[max_weight_idx].P)
            except np.linalg.LinAlgError:
                logging.error(f"Linalg error \n{states[max_weight_idx].P}")
                logging.warning("Get pseudo inverse matrix")
                S = np.linalg.pinv(states[max_weight_idx].P)

            idx_to_merge = []
            indices = list(range(len(states)))
            for idx in indices:
                state = states[idx]
                # Find Mahalanobis distance between state with max weight and current state
                x_diff = state.x - states[max_weight_idx].x
                mahalanobis_distance = x_diff.T @ S @ x_diff

                if mahalanobis_distance < threshold:
                    idx_to_merge.append(idx)

            # perform moment matching for states that close to state with max weights
            normalized_weights, log_sum_w = normalize_log_weights([weights[idx] for idx in idx_to_merge])
            merged_state = GaussianDensity.moment_matching(weights=normalized_weights, states=[states[idx] for idx in idx_to_merge])

            # Remove merged states from original list of states
            for idx in reversed(idx_to_merge):
                del states[idx]
                del weights[idx]

            new_log_weights.append(log_sum_w)
            new_merged_states.append(merged_state)

        return new_log_weights, new_merged_states
