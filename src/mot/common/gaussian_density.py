from typing import List

import numpy as np
from scipy.stats import multivariate_normal
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from mot.common.normalize_log_weights import normalize_log_weights
from mot.common.state import Gaussian
import numpy.typing as npt
import logging
import copy

logger = logging.getLogger()


class GaussianDensity:
    def __init__(self, parameter_list):
        """
        docstring
        """

    @staticmethod
    def predict(state: Gaussian, motion_model: MotionModel, dt: float) -> Gaussian:
        """Performs linear/nonlinear (Extended) Kalman prediction step

        Args:
            state (Gaussian): a structure with two fields - mean and covariance
            motion_model (MotionModel): a structure specifies the motion model parameters

        Returns:
            state_pred (Gaussian): predicted value of state
        """
        next_x = motion_model.f(state.x, dt)
        next_F = motion_model.F(state.x, dt)
        next_P = np.linalg.multi_dot([next_F, state.P, next_F.T]) + motion_model.Q(dt)
        return Gaussian(next_x, next_P)

    @staticmethod
    def update(
        state_pred: Gaussian, z, measurement_model: MeasurementModel
    ) -> Gaussian:
        """
        Performs linear/nonlinear (Extended) Kalman update step

        Args:
            z (Gaussian): a structure with two fields - mean and covariance
            measurement_model (MeasurementModel): a structure specifies
                                                  the measurement model parameters

        Returns:

            state_upd (Gaussian): updated state
        """
        if z.size == 0:
            logging.error("z size 0")
            return state_pred
        assert isinstance(state_pred, Gaussian)
        if z.ndim == 1:
            z = np.expand_dims(z, axis=0)

        assert z.shape[1] == measurement_model.dim

        # Measurement model Jacobian
        H_x = measurement_model.H(state_pred.x)

        # Innovation covariance
        S = np.linalg.multi_dot([H_x, state_pred.P, H_x.T]) + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + S.T)

        K = state_pred.P @ H_x.T @ np.linalg.inv(S)

        next_x = (
            state_pred.x + (K @ (z - measurement_model.h(state_pred.x)).T).squeeze()
        )

        assert next_x.shape == state_pred.x.shape

        # Covariance update
        # TODO ndim property for state
        I = np.eye(state_pred.x.shape[0])
        next_P = (I - K @ H_x) @ state_pred.P
        state_upd = Gaussian(x=next_x, P=next_P)
        return state_upd

    @staticmethod
    def predicted_likelihood(
        state_pred: Gaussian, z: npt.ArrayLike, measurement_model: MeasurementModel
    ) -> np.ndarray:
        """Calculates the predicted likelihood in logarithm domain

        Args:
            state_pred (Gaussian):
            z (np.ndarray (measurement dimension) x (number of measurements)): measurements
            measurement_model (MeasurementModel): specifies the measurement model

        Returns:
            predicted_likelihood (np.ndarray (number of measurements)): predicted likelihood
                                                                        for each measurement
                                                                        in logarithmic scale
        """

        assert z.ndim == 2
        assert z.shape[1] == measurement_model.dim
        # Measurement model Jacobian (z_bar)
        H_x = measurement_model.H(state_pred.x)
        # Innovation covariance
        S = H_x @ state_pred.P @ H_x.T + measurement_model.R
        # Make sure matrix S is positive definite
        S = (S + S.T) / 2

        # Predicted measurement
        z_bar = H_x @ state_pred.x

        predicted_likelihood = np.zeros(z.shape[0])  # size - num of measurements

        for idx in range(z.shape[0]):
            predicted_likelihood[idx] = multivariate_normal.logpdf(z[idx], z_bar, S)

        assert predicted_likelihood.shape[0] == z.shape[0]
        return predicted_likelihood

    @staticmethod
    def ellipsoidal_gating(
        state_prev: Gaussian,
        z: npt.ArrayLike,
        measurement_model: MeasurementModel,
        gating_size: float,
    ) -> np.ndarray:
        """Performs ellipsoidal gating for a single object

        Args:
            state_pred (Gaussian): predicted state
            z (np.ndarray (measurements dimenstion) x (number of measurements)): [description]
            measurement_model (MeasurementModel): specifies the measurement model parameters
            gating_size (float): gating size

        Returns:
            z_ingate (np.ndarray (measuremens dim) x (num of measurements in the gate))
                                                         : measurements in the gate
            meas_in_gate (List (num of measurements x 1)): boolean vector indicating whether
                                                           the corresponding measurement is
                                                           in the gate or not
        """
        if z.size == 0:
            logger.warning("No measurements! No updates!")
            return np.array([]), None
        assert z.shape[1] == measurement_model.dim

        # Measurements model Jacobian
        H_x = measurement_model.H(state_prev.x)

        # Innovation covariance
        S = H_x @ state_prev.P @ H_x.T

        # Make sure matrix S is positive definite
        S = (S + S.T) / 2

        # Predicted measurement
        z_bar = H_x @ state_prev.x

        # Difference between measurement and prediction
        z_diff = z - z_bar

        num_of_measurements = z.shape[0]

        # Squared Mahalanobis distance
        Machlanobis_dist = np.zeros(num_of_measurements)

        # TODO vectorize this
        for i in range(num_of_measurements):
            try:
                Machlanobis_dist[i] = z_diff[i] @ np.linalg.inv(S) @ z_diff[i].T
            except:
                logging.warning("It seems, cannot inverse S, skip step")
                return np.array([]), []
        # Machlanobis_dist = mln(z, z_bar, S) # z_diff @ np.linalg.inv(S) @ z_diff.Tx

        indices_in_gate = Machlanobis_dist < gating_size
        assert Machlanobis_dist.shape[0] == z.shape[0]
        z_ingate = z[indices_in_gate]
        return z_ingate, indices_in_gate

    @staticmethod
    def moment_matching(weights: List[float], states: List[Gaussian]) -> Gaussian:
        """Aproximates a Gaussian mixture density as a single Gaussian using moment matching

        Args:
            weights (List(float)): normalized weight of Gaussian components
                                   in logarithm domain
            states (List(Gaussian)): list of Gaussians

        Returns:
            Gaussian: resulted mixture
        """
        if len(weights) == 0:
            return

        log_weights = np.exp(weights)

        # Mean of the states
        N = len(weights)
        x_bar = np.zeros(states[0].x.shape[0])

        for idx, state, weight in zip(range(N), states, log_weights):
            x_bar += weight * state.x

        # Covariance
        P_bar = np.zeros_like(states[0].P)
        for idx in range(N):
            d = x_bar - states[idx].x
            P_bar += (states[idx].P + d @ d.T) * log_weights[idx]

        matched_state = Gaussian(x=x_bar, P=P_bar)
        return matched_state

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
            except:
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
            normalized_weights, log_sum_w = normalize_log_weights(
                [weights[idx] for idx in idx_to_merge]
            )
            merged_state = GaussianDensity.moment_matching(
                weights=normalized_weights, states=[states[idx] for idx in idx_to_merge]
            )

            # Remove merged states from original list of states
            for idx in reversed(idx_to_merge):
                del states[idx]
                del weights[idx]

            new_log_weights.append(log_sum_w)
            new_merged_states.append(merged_state)

        return new_log_weights, new_merged_states
