from copy import deepcopy
from typing import List, Tuple
import scipy
import numpy as np
from mot.common.gaussian_density import GaussianDensity
from mot.common.normalize_log_weights import normalize_log_weights
from mot.common.state import GaussianMixture
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from mot.trackers.multiple_object_trackers.PMBM.common.bernoulli import Bernoulli
from mot.trackers.multiple_object_trackers.PMBM.common.track import (
    SingleTargetHypothesis,
    Track,
)
import copy


class PoissonRFS:
    def __init__(self, intensity: GaussianMixture, *args, **kwargs):
        assert isinstance(intensity, GaussianMixture)
        self.intensity = deepcopy(intensity)

    def __repr__(self):
        return self.intensity.__repr__()

    def __len__(self):
        return len(self.intensity)

    def get_targets_detected_for_first_time(
        self,
        measurements: np.ndarray,
        gating_matrix_ud: np.ndarray,
        clutter_intensity: float,
        meas_model: MeasurementModel,
        detection_probability: float,
    ) -> List[Track]:

        new_tracks = {}
        for meas_idx in range(len(measurements)):
            if gating_matrix_ud.T[meas_idx].any():
                new_single_target_hypothesis = self.detected_update(
                    np.expand_dims(measurements[meas_idx], axis=0),
                    meas_model,
                    detection_probability,
                    clutter_intensity,
                    gating_vector=gating_matrix_ud.T[meas_idx],
                )
                new_track = Track.from_sth(new_single_target_hypothesis)
                new_track.single_target_hypotheses[0].meas_idx = meas_idx
                new_tracks[new_track.track_id] = new_track
        return new_tracks

    def detected_update(
        self,
        measurement: np.ndarray,
        meas_model: MeasurementModel,
        detection_probability: float,
        clutter_intensity: float,
        gating_vector,
        density=GaussianDensity,
    ) -> SingleTargetHypothesis:
        """Creates a new local hypothesis by updating the PPP
        with measurement and calculates the corresponding
        likelihood.

        z np .ndarray
            number of measurementx x measurement dimension

        Resulted Bernoulli component represents two possibilites:
        1) deteciton from clutter
        2) detection from a new object

        NOTE: p.152 in presentation

        """
        assert isinstance(meas_model, MeasurementModel)

        # 1. For each mixture component in the PPP intensity, perform Kalman update and
        # calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
        gated_ppp_components = [
            ppp_component for ppp_component_idx, ppp_component in enumerate(
                copy.deepcopy(self.intensity))
            if gating_vector[ppp_component_idx] == True
        ]
        updated_ppp_components = [
            GaussianDensity.update(ppp_component.gaussian, measurement,
                                   meas_model)
            for ppp_component in gated_ppp_components
        ]

        # Compute predicted likelihood
        log_weights = np.array([
            np.log(detection_probability) +
            ppp_component.log_weight + density.predict_loglikelihood(
                updated_state, measurement, meas_model).item()
            for ppp_component, updated_state in zip(self.intensity,
                                                    updated_ppp_components)
        ])

        # 2. Perform Gaussian moment matching for the updated object state densities
        # resulted from being updated by the same detection.
        normalized_log_weights, log_sum = normalize_log_weights(log_weights)
        merged_state = density.moment_matching(
            normalized_log_weights, updated_ppp_components
        )
        # merged_state = updated_ppp_components[0]

        # 3. The returned likelihood should be the sum of the predicted likelihoods calculated f
        # or each mixture component in the PPP intensity and the clutter intensity.
        # (You can make use of the normalizeLogWeights function to achieve this.)
        log_likelihood = scipy.special.logsumexp([log_sum, np.log(clutter_intensity)])

        # 4. The returned existence probability of the Bernoulli component
        # is the ratio between the sum of the predicted likelihoods
        # and the returned likelihood.
        # (Be careful that the returned existence probability is
        # in decimal scale while the likelihoods you calculated
        # beforehand are in logarithmic scale.)
        existence_probability = np.exp(log_sum - log_likelihood)
        bernoulli = Bernoulli(merged_state, existence_probability)
        cost = -log_likelihood
        return SingleTargetHypothesis(
            bernoulli=bernoulli,
            log_likelihood=log_likelihood,
            cost=cost,
            meas_idx=None,
            sth_id=0,
        )

    def undetected_update(self, detection_probability) -> None:
        """Performs PPP update for missed detection."""
        for ppp_component in self.intensity:
            ppp_component.log_weight += np.log(1 - detection_probability)

    def prune(self, threshold: float) -> None:
        self.intensity.weighted_components = [
            ppp_component
            for ppp_component in self.intensity.weighted_components
            if ppp_component.log_weight > threshold
        ]

    def predict(
        self,
        motion_model: MotionModel,
        survival_probability: float,
        density: GaussianDensity,
        dt: float,
    ) -> None:
        """Performs prediciton step for PPP components hypothesing undetected objects.
        Birth components will be added in another method."""
        assert isinstance(motion_model, MotionModel)
        assert isinstance(survival_probability, float)
        assert isinstance(dt, float)

        for ppp_component in self.intensity:
            ppp_component.log_weight += np.log(survival_probability)
            ppp_component.gaussian = density.predict(
                ppp_component.gaussian, motion_model, dt
            )

    def birth(self, new_components: GaussianMixture):
        """Incorporate PPP birth intensity into PPP intensity

        Parameters
        ----------
        born_components : GaussianMixture
            [description]
        """
        assert isinstance(new_components, GaussianMixture)
        self.intensity.extend(deepcopy(new_components))

    def gating(
        self,
        z: np.ndarray,
        density_handler,
        meas_model: MeasurementModel,
        gating_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns measurement indices inside the gate of undetected objects (PPP)"""
        gating_matrix_undetected = np.full(
            shape=[self.intensity.size, len(z)], fill_value=False
        )  # poisson size x number of measurements
        used_measurement_undetected_indices = np.full(shape=[len(z)], fill_value=False)

        for ppp_idx in range(self.intensity.size):
            (
                _,
                gating_matrix_undetected[ppp_idx],
            ) = density_handler.ellipsoidal_gating(
                self.intensity[ppp_idx].gaussian, z, meas_model, gating_size
            )
            used_measurement_undetected_indices = np.logical_or(
                used_measurement_undetected_indices, gating_matrix_undetected[ppp_idx]
            )
        return (gating_matrix_undetected, used_measurement_undetected_indices)
