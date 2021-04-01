import numpy as np
from copy import deepcopy

from mot.measurement_models import MeasurementModel
from mot.trackers.multiple_object_trackers.PMBM.common.bernoulli import Bernoulli
from mot.common.state import Gaussian, GaussianMixture
from mot.trackers.multiple_object_trackers.PMBM.common.track import (
    Track,
    SingleTargetHypothesis,
)
from typing import List, Tuple
from mot.common.gaussian_density import GaussianDensity
from mot.common.normalize_log_weights import normalize_log_weights
from mot.motion_models import MotionModel


class PoissonRFS:
    def __init__(self, initial_intensity=None, *args, **kwargs):
        self.intensity = deepcopy(initial_intensity)

    def __repr__(self):
        return self.intensity.__repr__()

    def get_targets_detected_for_first_time(
        self,
        z: np.ndarray,
        gating_matrix_ud: np.ndarray,
        clutter_intensity: float,
        meas_model: MeasurementModel,
        detection_probability: float,
    ) -> List[Track]:

        new_tracks = []
        dummy_bernoulli = Bernoulli(
            r=0.0, state=Gaussian(x=np.array([0, 0, 0, 0]), P=1000 * np.eye(4))
        )
        for meas_idx in range(len(z)):
            indices = gating_matrix_ud[
                :, meas_idx
            ]  # indices for PPP components gating with curr meas
            if indices.any():
                (new_target_bernoulli, new_target_likelihood,) = self.detected_update(
                    indices,
                    np.expand_dims(z[meas_idx], axis=0),
                    meas_model,
                    detection_probability,
                    clutter_intensity,
                )
            else:
                # There are already dummy bernoullis, no need to create some
                new_target_likelihood = np.log(clutter_intensity)
                new_target_bernoulli = dummy_bernoulli
            cost = -new_target_likelihood
            new_single_target_hypothesis = SingleTargetHypothesis(
                new_target_bernoulli, new_target_likelihood, meas_idx, cost
            )
            new_track = Track.from_sth(new_single_target_hypothesis)
            new_tracks.append(new_track)
        return new_tracks

    def detected_update(
        self,
        indices,
        z: np.ndarray,
        meas_model: MeasurementModel,
        detection_probability: float,
        clutter_intensity: float,
        density=GaussianDensity,
    ):
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

        # 1. For each mixture component in the PPP intensity, perform Kalman update and
        # calculate the predicted likelihood for each detection inside the corresponding ellipsoidal gate.
        clutter_intensity_log = np.log(clutter_intensity)

        P_D_log = np.log(detection_probability)

        PPP_gated_indices = []
        for idx, is_in_gate in enumerate(indices):
            if is_in_gate:
                PPP_gated_indices.append(idx)

        ppp_states_updated = [
            density.update(self.intensity[idx].gm, z, meas_model)
            for idx in PPP_gated_indices
        ]

        # Compute predicted likelihood
        likelihoods = [
            density.predicted_likelihood(state, z, meas_model).item()
            for state in ppp_states_updated
        ]

        # 2. Perform Gaussian moment matching for the updated object state densities
        # resulted from being updated by the same detection.
        weights_tilde_log = [
            likelihood + P_D_log + self.intensity[idx].w
            for likelihood, idx in zip(likelihoods, PPP_gated_indices)
        ]

        weights_log, log_sum = normalize_log_weights(weights_tilde_log)
        merged_state = density.moment_matching(weights_log, ppp_states_updated)
        # 3. The returned likelihood should be the sum of the predicted likelihoods calculated f
        # or each mixture component in the PPP intensity and the clutter intensity.
        # (You can make use of the normalizeLogWeights function to achieve this.)
        _, likelihood_new = normalize_log_weights([log_sum, clutter_intensity_log])

        # 4. The returned existence probability of the Bernoulli component
        # is the ratio between the sum of the predicted likelihoods
        # and the returned likelihood.
        # (Be careful that the returned existence probability is
        # in decimal scale while the likelihoods you calculated
        # beforehand are in logarithmic scale.)
        bernoulli = Bernoulli(
            r=np.exp(log_sum - likelihood_new), state=merged_state
        )
        return bernoulli, likelihood_new

    def undetected_update(self, detection_probability) -> None:
        """Performs PPP update for missed detection.

        Raises
        ------
        NotImplementedError
            [description]
        """
        self.intensity.weights += np.log(1 - detection_probability)

    def prune(self, threshold: float) -> None:
        self.intensity.weighted_components = [
            component
            for component in self.intensity.weighted_components
            if component.w > threshold
        ]

    def predict(
        self,
        motion_model: MotionModel = None,
        probability_survival: float = None,
        dt: float = 1.0,
        density=GaussianDensity,
    ) -> None:
        """Performs prediciton step for PPP components hypothesing undetected objects.
        Birth components will be added in another method."""

        # Predict Poisson intensity for pre-existing objects.
        for idx in range(len(self.intensity)):
            self.intensity[idx].log_weight += np.log(probability_survival)
            self.intensity[idx].gaussian = density.predict(
                state=self.intensity[idx].gaussian, motion_model=motion_model, dt=dt
            )

    def birth(self, born_components: GaussianMixture):
        """Incorporate PPP birth intensity into PPP intensity

        Parameters
        ----------
        born_components : GaussianMixture
            [description]
        """
        self.intensity.extend(deepcopy(born_components))

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
                self.intensity[ppp_idx].gm, z, meas_model, gating_size
            )
            used_measurement_undetected_indices = np.logical_or(
                used_measurement_undetected_indices, gating_matrix_undetected[ppp_idx]
            )
        return (gating_matrix_undetected, used_measurement_undetected_indices)
