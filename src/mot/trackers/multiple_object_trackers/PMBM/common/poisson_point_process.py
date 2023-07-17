import logging as lg
from copy import deepcopy
from distutils.log import Log
from functools import partial
from typing import List, Tuple

import nptyping as npt
import numpy as np
import scipy

from mot.common.gaussian_density import GaussianDensity, make_SPD
from mot.common.normalize_log_weights import normalize_log_weights
from mot.common.state import Observation
from mot.measurement_models import MeasurementModel
from mot.motion_models import BaseMotionModel
from mot.utils.timer import Timer

from .bernoulli import Bernoulli
from .single_target_hypothesis import SingleTargetHypothesis
from .track import Track


class PoissonRFS:
    def __init__(self, intensity: GaussianDensity):
        self.intensity = deepcopy(intensity)
        # self.pool = Pool(nodes=8)

    def __repr__(self):
        return self.intensity.__repr__()

    def __len__(self):
        return len(self.intensity)

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

        z np .ndarray
            number of measurementx x measurement dimension

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

        result, dists = GaussianDensity.ellipsoidal_gating(self.intensity, measurements, model_measurement, gating_size, H_x, S)
        # mask = ((np.exp(dists) / np.sum(np.exp(dists), axis=0)) < 0.9).T  # [n_measurements, n_gaussians]
        new_single_target_hypotheses = []

        for meas_idx, curr_mask in zip(range(len(measurements)), result.T):
            if np.sum(curr_mask) == False:
                continue
            # For each mixture component in the PPP intensity, perform Kalman update and calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
            next_state_means, next_state_covs, H_X_S = GaussianDensity.update(
                GaussianDensity(self.intensity.means[curr_mask], self.intensity.covs[curr_mask], self.intensity.weigths[curr_mask]),
                measurements[meas_idx][None, :],
                model_measurement,
            )

            updated_ppp_components = GaussianDensity(next_state_means, next_state_covs)
            # Compute predicted likelihood
            log_likelihoods: npt.NDArray[npt.Shape["N_gated_gaussians, N_measurements"], npt.Float] = GaussianDensity.predict_loglikelihood(
                GaussianDensity(self.intensity.means[curr_mask], self.intensity.covs[curr_mask], self.intensity.weigths_log[curr_mask]),
                measurements[meas_idx][None, :],
                model_measurement,
                H_X_S,
            )

            log_weights: npt.NDArray[npt.Shape["N_gated_gaussians, N_measurements"], npt.Float] = (
                log_likelihoods + self.intensity.weigths_log[curr_mask][:, None] + np.log(detection_probability)
            )  # same shape as loglikelihoods

            normalized_log_weights, log_sum = normalize_log_weights(log_weights, axis=-1)

            # Perform GaussianDensity moment matching for the updated object state densities resulted from being updated by the same detection.
            merged_means: npt.NDArray[npt.Shape["N_measurements, State_dim"], npt.Float] = np.zeros(
                (len(measurements), self.intensity.means.shape[-1])
            )
            merged_covs: npt.NDArray[npt.Shape["N_measurements, State_dim, State_dim"], npt.Float] = np.zeros(
                (len(measurements), self.intensity.covs.shape[-1], self.intensity.covs.shape[-1])
            )
            merge_state = GaussianDensity.moment_matching(
                normalized_log_weights[:, 0], GaussianDensity(next_state_means[:, 0, ...], next_state_covs[:, 0, ...])
            )

            # The returned likelihood should be the sum of the predicted likelihoods calculated for each mixture component in the PPP intensity and the clutter intensity. (You can make use of the normalizeLogWeights function to achieve this.)
            log_likelihood = scipy.special.logsumexp((log_sum, np.log(clutter_intensity)))

            # The returned existence probability of the Bernoulli component is the ratio between the sum of the predicted likelihoods and the returned likelihood. (Be careful that the returned existence probability is in decimal scale while the likelihoods you calculated beforehand are in logarithmic scale)
            existence_probability = np.exp(log_sum - log_likelihood)
            merge_state = GaussianDensity(merge_state.means[None, ...], merge_state.covs[None, ...])
            sth = SingleTargetHypothesis(
                bernoulli=Bernoulli(merge_state, existence_probability),
                log_likelihood=log_likelihood,
                cost=-log_likelihood,
                meas_idx=meas_idx,
                sth_id=0,
            )

            new_single_target_hypotheses.append(sth)
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
