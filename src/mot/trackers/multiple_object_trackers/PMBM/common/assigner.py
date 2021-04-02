import numpy as np
from mot.trackers.multiple_object_trackers.PMBM.common.global_hypothesis import (
    GlobalHypothesis, )
from mot.trackers.multiple_object_trackers.PMBM.common.global_hypothesis import (
    Association, )
from murty import Murty
import logging as lg
from collections import defaultdict
from typing import List


class AssignmentSolver:
    def __init__(
        self,
        global_hypothesis,
        old_tracks,
        new_tracks,
        measurements,
        num_of_desired_hypotheses,
    ) -> None:
        self.global_hypothesis: GlobalHypothesis = global_hypothesis
        self.old_tracks = old_tracks
        self.new_tracks = new_tracks
        self.measurements = measurements
        self.num_of_desired_hypotheses = num_of_desired_hypotheses
        self.num_of_old_tracks = len(self.global_hypothesis.associations)
        self.max_murty_steps = 10
        self.column_row_to_detected_child_sth = defaultdict(
            lambda: defaultdict(dict))
        self.column_row_to_new_detected_sth = defaultdict(
            lambda: defaultdict(dict))

    def get_murty_steps(self):
        return int(
            np.ceil(
                np.exp(self.global_hypothesis.log_weight *
                       self.num_of_desired_hypotheses)))

    def solve(self) -> List[GlobalHypothesis]:
        new_global_hypotheses = []
        cost_detected = self.create_cost_for_associated_targets(
            self.global_hypothesis, self.old_tracks, self.measurements)
        cost_undetected = self.create_cost_for_undetected(
            self.new_tracks, self.measurements)
        cost_matrix = np.hstack([cost_detected, cost_undetected])

        murty_solver = Murty(cost_matrix)

        for murty_step in range(self.max_murty_steps):
            try:
                status, solution_cost, solution = murty_solver.draw()
                # solution looks like array of assigned colums for meas idx
                solution = solution.tolist()
                if not solution:
                    break
                lg.debug(f"\n Assignment = {solution} Cost = {solution_cost}")
            except:
                raise Error
            if not status:
                lg.info("Break! Murty solver is over!")
                break
            next_log_weight = self.global_hypothesis.log_weight - solution_cost
            next_associations = self.assignment_to_associations(solution)
            next_global_hypothesis = GlobalHypothesis(next_log_weight,
                                                      next_associations)
            new_global_hypotheses.append(next_global_hypothesis)
        return new_global_hypotheses

    def assignment_to_associations(self, solution):
        associations = []
        for measurement_row, target_column in enumerate(solution):
            if target_column + 1 > self.num_of_old_tracks:
                # assignment is to new target
                track_id, sth_id = self.column_row_to_new_detected_sth[
                    target_column - self.num_of_old_tracks]
            else:
                # assignment is to a previously detected target
                (
                    track_id,
                    parent_sth_id,
                    child_idx,
                ) = self.column_row_to_detected_child_sth[target_column][
                    measurement_row]
                sth_id = (self.old_tracks[track_id].single_target_hypotheses[
                    parent_sth_id].detection_hypotheses[child_idx].sth_id)
            associations.append(Association(track_id, sth_id))
            return associations

    def create_cost_for_associated_targets(self,
                                           global_hypothesis: GlobalHypothesis,
                                           old_tracks,
                                           measurements) -> np.ndarray:
        cost_detected = np.full(
            (len(measurements), len(global_hypothesis.associations)), np.inf)
        for column_idx, (track_idx, parent_sth_idx) in enumerate(
                global_hypothesis.associations):
            parent_sth = old_tracks[track_idx].single_target_hypotheses[
                parent_sth_idx]
            for meas_idx, sth in parent_sth.detection_hypotheses.items():
                cost_detected[meas_idx, column_idx] = sth.cost
                self.column_row_to_detected_child_sth[column_idx][meas_idx] = (
                    track_idx,
                    parent_sth_idx,
                    meas_idx,
                )
        return cost_detected

    def create_cost_for_undetected(self, new_tracks,
                                   measurements) -> np.ndarray:
        # Using association between measurements and previously undetected objects

        cost_undetected = np.full((len(measurements), len(measurements)),
                                  np.inf)
        assert len(measurements) == len(new_tracks)

        sth_idx = 0  # we have olny one sth for new targets
        for meas_idx, track_idx in zip(range(len(measurements)),
                                       new_tracks.keys()):
            cost_undetected[meas_idx, meas_idx] = (
                new_tracks[track_idx].single_target_hypotheses[sth_idx].cost)
            self.column_row_to_new_detected_sth[meas_idx] = (
                new_tracks[track_idx].track_id,
                sth_idx,
            )
        return cost_undetected
