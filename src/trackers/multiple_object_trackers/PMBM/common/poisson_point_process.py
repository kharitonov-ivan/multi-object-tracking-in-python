import typing as tp
from copy import deepcopy, copy

import numpy as np
import scipy

from src.common.gaussian_density import GaussianDensity, make_SPD
from src.common.normalize_log_weights import normalize_log_weights
from src.measurement_models import MeasurementModel
from src.motion_models import BaseMotionModel

from .bernoulli import Bernoulli, SingleTargetHypothesis
from .multi_bernoulli_mixture import Track


import numpy as np
import scipy
import scipy.stats


# def vectorized_gaussian_logpdf(X, means, covariances):
#     """
#     Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
#     https://stackoverflow.com/a/50809036/8395138
#     Args:
#         X : shape (n, d)
#             Data points
#         means : shape (n, d)
#             Mean vectors
#         covariances : shape (n, d)
#             Diagonal covariance matrices
#     Returns:
#         logpdfs : shape (n,)
#             Log probabilities
#     """
#     _, d = X.shape
#     constant = d * np.log(2 * np.pi)
#     log_determinants = np.log(np.prod(covariances, axis=1))
#     deviations = X - means
#     inverses = 1 / covariances
#     return -0.5 * (constant + log_determinants + np.sum(deviations * inverses * deviations, axis=1))


def update_states_with_likelihoods_by_single_measurement(
        initial_states: GaussianDensity,
        measurement: np.ndarray,
        measurement_model: MeasurementModel,
    ):

        H_x = measurement_model.H(initial_states.means)
        # Innovation covariance
        H_x_T = np.transpose(H_x, axes=(0, 2, 1))
        S = H_x @ initial_states.covs @ H_x_T + measurement_model.R

        # Make sure matrix S is positive definite
        S = 0.5 * (S + np.transpose(S, axes=(0, 2, 1)))

        K = initial_states.covs @ H_x_T @ np.linalg.inv(S)

        measurement_row = np.vstack([measurement] * initial_states.size)
        fraction = measurement_row - measurement_model.h(initial_states.means)
        with_K = np.einsum("ijk,ik->ij", K, fraction)
        new_states = initial_states.means + with_K

        state_vector_size = initial_states.means[0].shape[0]

        next_covariances = (np.eye(state_vector_size) - K @ H_x) @ initial_states.covs

        next_states = [GaussianDensity(new_states[idx], next_covariances[idx]) for idx in range(initial_states.size)]

        measurements_bar = H_x @ initial_states.means.T

        # it takes 0.0277 sec
        loglikelihoods = [
            scipy.stats.multivariate_normal.logpdf(measurement, measurements_bar[idx], S[idx])
            for idx in range(initial_states.size)
        ]

        # loglikelihoods_fast = vectorized_gaussian_logpdf(
        #     X=measurement_row,
        #     means=measurements_bar.squeeze().T,
        #     covariances=np.diagonal(S, axis1=2),
        # )

        # np.testing.assert_almost_equal(loglikelihoods, loglikelihoods_fast)

        return next_states, loglikelihoods#, loglikelihoods_fast



class PoissonRFS:
    def __init__(self, intensity: GaussianDensity):
        self.intensity = deepcopy(intensity)

    def get_targets_detected_for_first_time(
        self,
        measurements: np.ndarray,
        clutter_intensity: float,
        model_measurement: MeasurementModel,
        detection_probability: float,
        gating_size: float,
    ) -> dict[int, Track]:
        """Creates a new local hypothesis by updating the PPP
        with measurement and calculates the corresponding
        likelihood.

        Resulting Bernoulli component represents two possibilites:
        1) deteciton from clutter
        2) detection from a new object

        NOTE: p.152 in presentation
        """
        # Elipse gating : Returns measurement indices inside the gate of undetected objects (PPP)"""

        # Reuse innovation covs
        H_x = model_measurement.H(self.intensity.means)  # Measurement model Jacobian
        H_x_T = np.swapaxes(H_x, -1, -2)
        S = H_x @ self.intensity.covs @ H_x_T + model_measurement.R  # Innovation covariance
        S = make_SPD(S)  # Make sure matrix S is positive definite

        result, dists = GaussianDensity.ellipsoidal_gating(self.intensity, measurements, model_measurement, gating_size)
        # import pdb; pdb.set_trace()
        result = np.ones_like(result, dtype=bool)
        # mask = ((np.exp(dists) / np.sum(np.exp(dists), axis=0)) < 0.9).T  # [n_measurements, n_gaussians]
        new_single_target_hypotheses = []

        for meas_idx, curr_mask in zip(range(len(measurements)), result.T):
            if np.sum(curr_mask) == False:
                continue
            # For each mixture component in the PPP intensity, perform Kalman update and calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
            


            # import pdb; pdb.set_trace()
            next_state_means, next_state_covs, _ = GaussianDensity.update(
                GaussianDensity(
                    self.intensity.means[curr_mask],
                    self.intensity.covs[curr_mask],
                    self.intensity.weights[curr_mask],
                ),
                measurements[meas_idx][None, :],
                model_measurement,
            )
            # import pdb; pdb.set_trace()

            next_state_means, next_state_covs = (
                next_state_means[:, 0, ...],
                next_state_covs[:, 0, ...],
            )
            updated_ppp_components = GaussianDensity(next_state_means, next_state_covs)
            # Compute predicted likelihood
            log_likelihoods: tp.Annotated[np.ndarray, "n_gated_gaussians, n_measurements"] = GaussianDensity.predict_loglikelihood(
                GaussianDensity(
                    self.intensity.means[curr_mask],
                    self.intensity.covs[curr_mask],
                    self.intensity.weights[curr_mask],
                ),
                measurements[meas_idx][None, :],
                model_measurement,
                None,
            )

            # next_states, loglikelihoods_fast = update_states_with_likelihoods_by_single_measurement(self.intensity, measurements[meas_idx], model_measurement)
            # import pdb; pdb.set_trace()

            log_weights: tp.Annotated[np.ndarray, "N_gated_gaussians, N_measurements"] = (
                log_likelihoods + self.intensity.weights[curr_mask][:, None] + np.log(detection_probability)
            )  # same shape as loglikelihoods
            normalized_log_weights, log_sum = normalize_log_weights(log_weights[..., 0], axis=-1)

            # Perform GaussianDensity moment matching for the updated object state densities resulted from being updated by the same detection.
            merged_means: tp.Annotated[np.ndarray, "N_measurements, State_dim"] = np.zeros((len(measurements), self.intensity.means.shape[-1]))
            merged_covs: tp.Annotated[np.ndarray, "(n_measurements, state_dim, state_dim)"] = np.zeros(
                (
                    len(measurements),
                    self.intensity.covs.shape[-1],
                    self.intensity.covs.shape[-1],
                )
            )
            merge_state = GaussianDensity.moment_matching(
                normalized_log_weights,
                GaussianDensity(next_state_means, next_state_covs),
            )

            # The returned likelihood should be the sum of the predicted likelihoods calculated for each mixture component in the PPP intensity and the clutter intensity.
            # (You can make use of the normalizeLogWeights function to achieve this.)
            log_likelihood = scipy.special.logsumexp((log_sum, np.log(clutter_intensity)))

            # The returned existence probability of the Bernoulli component is the ratio between the sum of the predicted likelihoods and the returned likelihood. (Be careful that the returned existence probability is in decimal scale while the likelihoods you calculated beforehand are in logarithmic scale)
            existence_probability = np.exp(log_sum - log_likelihood)
            merge_state = GaussianDensity(merge_state.means, merge_state.covs)
            sth = SingleTargetHypothesis(
                bernoulli=Bernoulli(merge_state, existence_probability),
                log_likelihood=log_likelihood,
                cost=-log_likelihood,
                meas_idx=meas_idx,
                sth_id=0,
            )

            new_single_target_hypotheses.append(sth)
        # import pdb; pdb.set_trace()
        # lg.error(f"{len(measurements)}, {len(new_single_target_hypotheses)}")
        new_tracks = {}
        for sth_ in new_single_target_hypotheses:
            new_track = Track.from_sth(sth_)
            new_tracks[new_track.track_id] = new_track
        return new_tracks

    def undetected_update(self, detection_probability) -> None:
        """Performs PPP update for missed detection."""
        self.intensity.weights = self.intensity.weights + np.log(1 - detection_probability)

    def prune(self, threshold: float) -> None:
        mask_to_keep = self.intensity.weights > threshold
        self.intensity.means, self.intensity.covs, self.intensity.weights = (
            self.intensity.means[mask_to_keep],
            self.intensity.covs[mask_to_keep],
            self.intensity.weights[mask_to_keep],
        )

    def predict(
        self,
        motion_model: BaseMotionModel,
        survival_probability: float,
        density: GaussianDensity,
        dt: float,
    ) -> None:
        """Performs prediciton step for PPP components hypothesing undetected objects.
        Birth components will be added in another method."""
        self.intensity.weights += np.log(survival_probability)
        self.intensity = density.predict(self.intensity, motion_model, dt)

    def birth(self, new_components: GaussianDensity) -> None:
        """Incorporate PPP birth intensity into PPP intensity

        Parameters
        ----------
        born_components : GaussianDensity
            [description]
        """
        self.intensity += new_components
