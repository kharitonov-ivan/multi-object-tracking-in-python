import logging as lg
from copy import deepcopy
from functools import partial
from typing import List, Tuple

import nptyping as npt
import numpy as np
import scipy

from mot.common import (
    Gaussian,
    GaussianDensity,
    GaussianMixture,
    GaussianMixtureNumpy,
    Observation,
    normalize_log_weights,
)
from mot.measurement_models import MeasurementModel
from mot.motion_models import BaseMotionModel
from mot.utils.timer import Timer

from .bernoulli import Bernoulli
from .single_target_hypothesis import SingleTargetHypothesis
from .track import Track


class PoissonRFS:
    def __init__(self, intensity: GaussianMixtureNumpy):
        # self.intensity = deepcopy(intensity)
        self.intensity = intensity
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
        # Elipse gating

        # For each mixture component in the PPP intensity, perform Kalman update and calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
        next_state_means, next_state_covs, H_X_S = GaussianDensity.update(self.intensity, measurements, model_measurement)
        updated_ppp_components = Gaussian(next_state_means, next_state_covs)
        # Compute predicted likelihood
        log_likelihoods: npt.NDArray[npt.Shape["N_gaussians, N_measurements"], npt.Float] = GaussianDensity.predict_loglikelihood(
            self.intensity, measurements, model_measurement, H_X_S
        )
        assert len(self.intensity) == len(log_likelihoods)

        log_weights: npt.NDArray[npt.Shape["N_gaussians, N_measurements"], npt.Float] = (
            log_likelihoods + self.intensity.weigths_log[..., None] + np.log(detection_probability)
        )  # same shape as loglikelihoods

        normalized_log_weights, log_sum = np.zeros_like(log_weights), np.zeros(len(measurements))
        for i in range(len(measurements)):
            normalized_log_weights[..., i], log_sum[i] = normalize_log_weights(log_weights[..., i], axis=-1)

        # Perform Gaussian moment matching for the updated object state densities resulted from being updated by the same detection.
        merged_means: npt.NDArray[npt.Shape["N_measurements, State_dim"], npt.Float] = np.zeros(
            (len(measurements), self.intensity.means.shape[-1])
        )
        merged_covs: npt.NDArray[npt.Shape["N_measurements, State_dim, State_dim"], npt.Float] = np.zeros(
            (len(measurements), self.intensity.covs.shape[-1], self.intensity.covs.shape[-1])
        )
        for meas_idx in range(len(measurements)):
            states = GaussianDensity.moment_matching(
                normalized_log_weights[..., meas_idx],
                Gaussian(updated_ppp_components.means[..., meas_idx, :], updated_ppp_components.covs[:, meas_idx, ...]),
            )
            merged_means[meas_idx], merged_covs[meas_idx] = states.means, states.covs

        merged_states = Gaussian(merged_means, merged_covs)  # [n_measurements, ...]

        # The returned likelihood should be the sum of the predicted likelihoods calculated for each mixture component in the PPP intensity and the clutter intensity. (You can make use of the normalizeLogWeights function to achieve this.)
        log_likelihood = np.array([scipy.special.logsumexp([log_sum_i, np.log(clutter_intensity)]) for log_sum_i in log_sum])

        # The returned existence probability of the Bernoulli component is the ratio between the sum of the predicted likelihoods and the returned likelihood. (Be careful that the returned existence probability is in decimal scale while the likelihoods you calculated beforehand are in logarithmic scale)
        existence_probability = np.exp(log_sum - log_likelihood)

        new_single_target_hypotheses = [
            SingleTargetHypothesis(
                bernoulli=Bernoulli(merged_states[meas_idx], existence_probability[meas_idx]),
                log_likelihood=log_likelihood[meas_idx],
                cost=-log_likelihood[meas_idx],
                meas_idx=meas_idx,
                sth_id=0,
            )
            for meas_idx in range(len(measurements))
        ]

        new_tracks = {}
        for meas_idx in range(len(measurements)):
            new_track = Track.from_sth(new_single_target_hypotheses[meas_idx])
            new_tracks[new_track.track_id] = new_track
        return new_tracks

    def undetected_update(self, detection_probability) -> None:
        """Performs PPP update for missed detection."""
        self.intensity.weigths_log += np.log(1 - detection_probability)

    def prune(self, threshold: float) -> None:
        mask_to_keep = np.squeeze(self.intensity.weigths_log > threshold)
        self.intensity.means, self.intensity.covs, self.intensity.weigths_log = (
            self.intensity.means[mask_to_keep],
            self.intensity.covs[mask_to_keep],
            self.intensity.weigths_log[mask_to_keep],
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
        self.intensity.weigths_log += np.log(survival_probability)
        self.intensity.means, self.intensity.covs = density.predict(self.intensity.means, self.intensity.covs, motion_model, dt)

    def birth(self, new_components: GaussianMixtureNumpy) -> None:
        """Incorporate PPP birth intensity into PPP intensity

        Parameters
        ----------
        born_components : GaussianMixture
            [description]
        """
        self.intensity += new_components
