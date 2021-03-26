from typing import List, Tuple
from mot.trackers.multiple_object_trackers.PMBM.common.bernoulli import Bernoulli
from collections import defaultdict
import numpy as np
from mot.common.normalize_log_weights import normalize_log_weights
from mot.trackers.multiple_object_trackers.PMBM.common.track import (
    Track,
    SingleTargetHypothesis,
)
from mot.common.estimation import Estimation
import logging


class MultiBernouilliMixture:
    """Track oriented approach using."""

    def __init__(self):
        self.tracks = {}
        self.global_hypotheses = []

    def add_track(self, track: Track):
        self.tracks.update({track.track_id: track})

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " "
            + (
                f"num of tracks= {len(self.tracks)}, "
                f"num global hypotheses={len(self.global_hypotheses)}, "
            )
        )

    def estimator(self):
        """Simply return objects set based on most probable global hypo."""
        if self.global_hypotheses:
            most_probable_global_hypo = max(
                self.global_hypotheses, key=lambda x: x.weight
            )
        else:
            logging.info("Pool of global hypotheses is empty!")
            return None

        object_list = []  # list with {'object_id':'object_state'}
        for (track_id, sth_id) in most_probable_global_hypo.hypothesis:
            object_state = (
                self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.state.x
            )
            object_list.append({track_id: object_state})
        return object_list

    def predict(self, motion_model, P_S, density_handler, dt) -> None:
        # MBM predict
        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                self.tracks[track_id].single_target_hypotheses[
                    sth_id
                ].bernoulli = Bernoulli.predict(
                    track.single_target_hypotheses[sth_id].bernoulli,
                    motion_model,
                    P_S,
                    density_handler,
                    dt,
                )

    def gating(self, z, density_handler, meas_model, gating_size):

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

    def update(self, P_D: float, z: np.ndarray, gating_matrix, meas_model) -> None:
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
                    likelihood_undetected,
                ) = Bernoulli.undetected_update(sth.bernoulli, P_D)

                missdetection_hypothesis = SingleTargetHypothesis(
                    bernoulli_undetected,
                    likelihood_undetected,
                    associated_measurement_idx=-1,
                    cost=None,
                )

                sth.children.update({-1: missdetection_hypothesis})

                likelihood_detected = Bernoulli.detected_update_likelihood(
                    sth.bernoulli,
                    z,
                    meas_model,
                    P_D,
                )
                # Create association hypothesis for measurements which in the gate
                for meas_idx, meas in enumerate(z):
                    if gating_matrix[track_id][sth_id][meas_idx]:
                        bernoulli_detected = Bernoulli.detected_update_state(
                            sth.bernoulli,
                            z[meas_idx],
                            meas_model,
                        )
                        cost = likelihood_detected[meas_idx] - likelihood_undetected
                        detection_hypothesis = SingleTargetHypothesis(
                            bernoulli_detected,
                            likelihood_detected[meas_idx],
                            meas_idx,
                            cost,
                        )
                        sth.children.update({meas_idx: detection_hypothesis})

    def prune_global_hypotheses(self, threshold=np.log(0.05)) -> None:
        """Removes Bernoulli components with small probability of existence and reindex the hypothesis table.
        If a track contains no single object hypothesis after pruning, this track is removed

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
        for global_hypo_idx, global_hypo in enumerate(self.global_hypotheses):
            self.global_hypotheses[global_hypo_idx].weight = global_hypo_log_w_norm[
                global_hypo_idx
            ]
