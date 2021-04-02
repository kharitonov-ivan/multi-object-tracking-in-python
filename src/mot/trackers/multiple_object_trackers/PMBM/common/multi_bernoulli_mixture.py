import logging
from collections import defaultdict
from typing import List

import numpy as np
from mot.common.normalize_log_weights import normalize_log_weights
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from mot.trackers.multiple_object_trackers.PMBM.common.track import Track

from .global_hypothesis import GlobalHypothesis


class MultiBernouilliMixture:
    """Track oriented approach using."""

    def __init__(self):
        self.tracks = {}
        self.global_hypotheses = []

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " "
            + (
                f"num of tracks= {len(self.tracks)}, "
                f"num global hypotheses={len(self.global_hypotheses)}, "
            )
        )

    def add_track(self, track: Track):
        assert track.track_id in self.tracks.keys()
        self.tracks[track.track_id] = track

    def estimator(self):
        """Simply return objects set based on most probable global hypo."""
        if self.global_hypotheses:
            most_probable_global_hypo = max(
                self.global_hypotheses, key=lambda x: x.weight
            )
        else:
            logging.info("Pool of global hypotheses is empty!")
            return None

        object_list = []  # list of {'object_id':'object_state'}
        for (track_id, sth_id) in most_probable_global_hypo.hypothesis:
            object_state = (
                self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.state.x
            )
            object_list.append({track_id: object_state})

        return object_list

    def predict(self, motion_model, survival_probability, density_handler, dt) -> None:
        # MBM predict
        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                self.tracks[track_id].single_target_hypotheses[
                    sth_id
                ].bernoulli = Bernoulli.predict(
                    track.single_target_hypotheses[sth_id].bernoulli,
                    motion_model,
                    survival_probability,
                    density_handler,
                    dt,
                )

    def gating(
        self, z: np.ndarray, density_handler, meas_model: MeasurementModel, gating_size
    ):

        gating_matrix = defaultdict(lambda: defaultdict(lambda: False))
        used_measurement_detected_indices = np.full(shape=[len(z)], fill_value=False)

        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                (
                    _,
                    gating_matrix[track_id][sth_id],
                ) = density_handler.ellipsoidal_gating(
                    sth.bernoulli.state, z, meas_model, gating_size
                )
                used_measurement_detected_indices = np.logical_or(
                    used_measurement_detected_indices,
                    gating_matrix[track_id][sth_id],
                )

        return (gating_matrix, used_measurement_detected_indices)

    def update(self, detection_probability: float, z: np.ndarray, gating_matrix, meas_model) -> None:
        """For each Bernoulli state density,
        create a misdetection hypothesis (Bernoulli component), and
        m object detection hypothesis (Bernoulli component),
        where m is the number of detections inside the ellipsoidal gate of the given state density.

        used_meas_indices : (hyp_tree_idx x bern_idx x meas_idx)
            [description]
        """

        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                sth.children = {}
                # Create missdetection hypothesis
                (
                    bernoulli_undetected,
                    log_likelihood_undetected,
                ) = sth.bernoulli.undetected_update(detection_probability)

                missdetection_hypothesis = SingleTargetHypothesis(
                    bernoulli_undetected,
                    log_likelihood_undetected,
                    associated_measurement_idx=-1,
                    cost=None,
                )
                logging.debug(
                    f"track id ={track_id}, sth_id = {sth_id}, {self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli}"
                )
                logging.debug(f"missdetection likelihood {log_likelihood_undetected}")
                sth.children.update({-1: missdetection_hypothesis})

                log_likelihoods_detected = Bernoulli.detected_update_likelihood(
                    sth.bernoulli,
                    z,
                    meas_model,
                    detection_probability,
                )
                # Create association hypothesis for measurements which in the gate
                for meas_idx, meas in enumerate(z):
                    bernoulli_detected = Bernoulli.detected_update_state(
                        sth.bernoulli,
                        z[meas_idx],
                        meas_model,
                    )
                    logging.debug(
                        f"detection z = {meas} likelihood {log_likelihoods_detected[meas_idx]}"
                    )
                    detection_hypothesis = SingleTargetHypothesis(
                        bernoulli_detected,
                        log_likelihoods_detected[meas_idx],
                        meas_idx,
                        cost=-(
                            log_likelihoods_detected[meas_idx]
                            - log_likelihood_undetected
                        ),
                    )
                    sth.children.update({meas_idx: detection_hypothesis})

    def prune_global_hypotheses(self, threshold) -> None:
        """Removes Bernoulli components with small probability of existence and reindex the hypothesis table.
        If a track contains no single object hypothesis after pruning, this track is removed

        good choice threshold : np.log(0.05)
        """
        self.global_hypotheses = [
            global_hypothesis
            for global_hypothesis in self.global_hypotheses
            if global_hypothesis.weight > threshold
        ]
        self.normalize_global_hypotheses_weights()

    def cap_global_hypothesis(self, max_number_of_global_hypothesis: int = 50):
        if len(self.global_hypotheses) > max_number_of_global_hypothesis:
            top_global_hypotheses = sorted(
                self.global_hypotheses, key=lambda x: x.weight
            )
            self.global_hypotheses = top_global_hypotheses[
                :max_number_of_global_hypothesis
            ]
            self.normalize_global_hypotheses_weights()

    def not_exist_of_bern_global_hypos(self, track_id, sth_id):
        result = True
        for global_hypothesis in self.global_hypotheses:
            if (track_id, sth_id) in global_hypothesis.hypothesis:
                result = False
        return result

    def remove_unused_bernoullies(self):
        list_to_remove = []
        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                if self.not_exist_of_bern_global_hypos(track_id, sth_id):
                    list_to_remove.append((track_id, sth_id))

        # for track_id, track in self.tracks.items():
        #     self.tracks[track_id].single_target_hypotheses = {
        #         k: v
        #         for k, v in track.single_target_hypotheses.items()
        #         if ((track_id, k) not in list_to_remove)
        #     }

        # # remove track without sth
        # self.tracks = {
        #     track_id: track
        #     for track_id, track in self.tracks.items()
        #     if len(track.single_target_hypotheses) != 0
        # }

    def normalize_global_hypotheses_weights(self) -> None:
        global_hypo_log_w_unnorm = [
            global_hypothesis.weight for global_hypothesis in self.global_hypotheses
        ]
        global_hypo_log_w_norm, _ = normalize_log_weights(global_hypo_log_w_unnorm)

        for global_hypo, normalized_log_weight in zip(
            self.global_hypotheses, global_hypo_log_w_norm
        ):
            global_hypo.weight = normalized_log_weight
