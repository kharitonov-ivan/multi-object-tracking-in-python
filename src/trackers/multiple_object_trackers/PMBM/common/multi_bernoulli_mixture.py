import logging
import pprint
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Mapping, NamedTuple, Tuple

import numpy as np

# TODO make it ok
from .....common.normalize_log_weights import normalize_log_weights
from .....measurement_models import MeasurementModel
from .....motion_models import BaseMotionModel
from .bernoulli import SingleTargetHypothesis


class Association(NamedTuple):
    track_id: int  # num of tree
    sth_id: int  # num of hypo in tree


@dataclass
class GlobalHypothesis:
    log_weight: float
    associations: Tuple[Association]

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(w={self.log_weight:.2f}, " f"(track_id, sth_id)={self.associations}, ")

    def __post_init__(self) -> None:
        if not self.associations:
            raise ValueError("Global hypothesis can not be without any association!")


class Track:
    """Represents a track - hypotheses tree.
    The root is associated with a unique target.
    Leaves are hypotheses which represent association of this target
    with corresponding measurements or misdetections.
    """

    max_track_id = 1000000
    current_idx = 0

    def get_id(cls):
        returned_idx = Track.current_idx
        Track.current_idx += 1
        return returned_idx

    def __init__(self, initial_sth=None):
        self.track_id = Track.get_id(Track)
        self.sth_id_counter = 0
        self.single_target_hypotheses = {self.get_new_sth_id(): initial_sth}

    def get_new_sth_id(self):
        returned_value = self.sth_id_counter
        self.sth_id_counter += 1
        return returned_value

    def add_sth(self, sth: SingleTargetHypothesis) -> None:
        self.single_target_hypotheses[self.get_new_sth_id()] = sth

    def __repr__(self) -> str:
        sth_rep = f"STH: \n {pprint.pformat(self.single_target_hypotheses)}" if len(self.single_target_hypotheses) < 3 else ""
        return self.__class__.__name__ + (f" id = {self.track_id}" f" number of sth = {len(self.single_target_hypotheses)} ") + sth_rep

    @classmethod
    def from_sth(cls, single_target_hypo):
        return cls(initial_sth=single_target_hypo)

    def cut_tree(self):
        new_single_target_hypotheses = {}
        for parent_sth in self.single_target_hypotheses.values():
            for child_sth in parent_sth.detection_hypotheses.values():
                new_single_target_hypotheses[child_sth.sth_id] = child_sth
        self.single_target_hypotheses = new_single_target_hypotheses


class MultiBernouilliMixture:
    """Track oriented approach using."""

    def __init__(self):
        self.tracks: Mapping[int, Track] = {}  # track_id -> Track
        self.global_hypotheses: List[GlobalHypothesis] = []  # heapify

    def __repr__(self) -> str:
        return self.__class__.__name__ + " " + (f"num of tracks= {len(self.tracks)}, " f"num global hypotheses={len(self.global_hypotheses)}, ")

    def add_track(self, track: Track):
        assert not (track.track_id in self.tracks.keys())
        self.tracks[track.track_id] = track

    def estimator(self, existense_probability_threshold, track_history_length_threshold):
        """Simply return objects set based on most probable global hypo."""
        if not self.global_hypotheses:
            logging.debug("Pool of global hypotheses is empty!")
            return {}

        most_probable_global_hypo = max(self.global_hypotheses, key=lambda x: x.log_weight)
        logging.debug(f"most probable global hypothesis: {most_probable_global_hypo}")

        objects = {}  #  {'object_id':'object_state'}
        logging.debug("\n estimations")
        for track_id, sth_id in most_probable_global_hypo.associations:
            if self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.existence_probability > existense_probability_threshold and sth_id > track_history_length_threshold:
                objects[track_id] = self.tracks[track_id].single_target_hypotheses[sth_id].bernoulli.state
        return objects

    def predict(
        self,
        motion_model: BaseMotionModel,
        survival_probability: float,
        density_handler,
        dt: float,
    ) -> None:
        # MBM predict
        for track in self.tracks.values():
            for sth in track.single_target_hypotheses.values():
                sth.bernoulli.predict(
                    motion_model,
                    survival_probability,
                    density_handler,
                    dt,
                )

    def gating(self, z: np.ndarray, density_handler, meas_model: MeasurementModel, gating_size):
        gating_matrix = defaultdict(lambda: defaultdict(lambda: False))
        used_measurement_detected_indices = np.full(shape=[len(z)], fill_value=False)

        for track_id, track in self.tracks.items():
            for sth_id, sth in track.single_target_hypotheses.items():
                gating_matrix[track_id][sth_id][1] = density_handler.ellipsoidal_gating(sth.bernoulli.state, z, meas_model, gating_size)
                used_measurement_detected_indices = np.logical_or(
                    used_measurement_detected_indices,
                    gating_matrix[track_id][sth_id],
                )

        return (gating_matrix, used_measurement_detected_indices)

    def update(
        self,
        detection_probability: float,
        measurements: np.ndarray,
        meas_model,
        density,
    ) -> None:
        """For each Bernoulli state density,
        create a misdetection hypothesis (Bernoulli component), and
        m object detection hypothesis (Bernoulli component),
        where m is the number of detections inside the ellipsoidal gate of the given state density.

        used_meas_indices : (hyp_tree_idx x bern_idx x meas_idx)
            [description]
        """
        logging.debug("\n Creating new STH in MBM")
        for track in self.tracks.values():
            for sth in track.single_target_hypotheses.values():
                logging.debug(f"\n For hypothesis: track_id={track.track_id} sth_id={sth.sth_id} {sth}")
                sth.missdetection_hypothesis = sth.create_missdetection_hypothesis(detection_probability, track.get_new_sth_id())

                detection_hypos = sth.create_detection_hypotheses(  # mapping int -> STH
                    measurements,
                    detection_probability,
                    meas_model,
                    density,
                    [track.get_new_sth_id() for i in range(len(measurements))],
                )
                sth.detection_hypotheses = detection_hypos
        logging.debug("\n Creating new STH in MBM: Done")

    def prune_global_hypotheses(self, log_threshold: float) -> None:
        """Removes Bernoulli components with small probability of existence and reindex the hypothesis table.
        If a track contains no single object hypothesis after pruning, this track is removed

        good choice threshold : np.log(0.05)
        """
        self.global_hypotheses = [global_hypothesis for global_hypothesis in self.global_hypotheses if global_hypothesis.log_weight > log_threshold]
        self.normalize_global_hypotheses_weights()

    def cap_global_hypothesis(self, max_number_of_global_hypothesis: int = 50):
        if len(self.global_hypotheses) > max_number_of_global_hypothesis:
            sorted_global_hypotheses = sorted(self.global_hypotheses, key=lambda x: x.log_weight)
            self.global_hypotheses = sorted_global_hypotheses[:max_number_of_global_hypothesis]
            self.normalize_global_hypotheses_weights()

    def prune_tree(self):
        used_associations = defaultdict(set)
        for global_hypothesis in self.global_hypotheses:
            for track_id, sth_id in global_hypothesis.associations:
                used_associations[track_id] |= set([sth_id])

        for track_id, track in self.tracks.items():
            track.single_target_hypotheses = {sth_id: track.single_target_hypotheses[sth_id] for sth_id in used_associations[track_id]}

    def remove_unused_tracks(self):
        used_associations = defaultdict(set)
        for global_hypothesis in self.global_hypotheses:
            for track_id, sth_id in global_hypothesis.associations:
                used_associations[track_id] |= set([sth_id])
        self.tracks = {track_id: track for (track_id, track) in self.tracks.items() if track_id in used_associations.keys()}

    def remove_unused_bernoullies(self):
        used_associations = defaultdict(set)
        for global_hypothesis in self.global_hypotheses:
            for track_id, sth_id in global_hypothesis.associations:
                used_associations[track_id] |= set([sth_id])
        for track_id, track in self.tracks.items():
            track.single_target_hypotheses = {sth_id: sth for (sth_id, sth) in track.single_target_hypotheses.items() if sth_id in used_associations[track_id]}

    def normalize_global_hypotheses_weights(self) -> None:
        # TODO make it ok in numpy
        if self.global_hypotheses:
            global_hypo_log_w_unnorm = np.array([global_hypothesis.log_weight for global_hypothesis in self.global_hypotheses])
            global_hypo_log_w_norm, _ = normalize_log_weights(global_hypo_log_w_unnorm)
            for global_hypo, normalized_log_weight in zip(self.global_hypotheses, global_hypo_log_w_norm):
                global_hypo.log_weight = normalized_log_weight
