import logging as lg
from collections import defaultdict
from typing import List

import numpy as np
from murty import Murty

from .global_hypothesis import Association, GlobalHypothesis


class AssignmentSolver:
    def __init__(
        self,
        global_hypothesis,
        old_tracks,
        new_tracks,
        measurements,
        num_of_desired_hypotheses,
        max_murty_steps=None,
    ) -> None:
        assert len(measurements) > 0
        self.global_hypothesis: GlobalHypothesis = global_hypothesis
        self.old_tracks = old_tracks
        self.new_tracks = new_tracks
        self.measurements = measurements
        self.num_of_desired_hypotheses = num_of_desired_hypotheses
        self.num_of_old_tracks = len(self.global_hypothesis.associations)
        self.max_murty_steps = max_murty_steps or self.get_murty_steps()
        self.column_row_to_detected_child_sth = defaultdict(lambda: defaultdict(dict))
        self.column_row_to_new_detected_sth = defaultdict(lambda: defaultdict(dict))
        self.cost_matrix = self.create_cost_matrix()
        if self.cost_matrix.size == 0:
            return

    def get_murty_steps(self):

        return int(
            np.ceil(
                np.exp(self.global_hypothesis.log_weight)
                * self.num_of_desired_hypotheses
            )
        )

    def create_cost_matrix(self):
        cost_detected = self.create_cost_for_associated_targets(
            self.global_hypothesis, self.old_tracks, self.measurements
        )
        cost_undetected = self.create_cost_for_undetected(
            self.new_tracks, self.measurements
        )
        cost_matrix = np.hstack([cost_detected, cost_undetected])
        return cost_matrix

    def solve(self) -> List[GlobalHypothesis]:
        lg.debug(f"\n Current global hypo = \n{self.global_hypothesis}")
        lg.debug(f"\n Cost matrix = \n{self.cost_matrix}")
        new_global_hypotheses = []
        murty_solver = Murty(self.cost_matrix)
        for murty_step in range(self.max_murty_steps):
            try:
                status, solution_cost, solution = murty_solver.draw()
                # solution looks like array of assigned colums for meas idx
                solution = solution.tolist()
                if not solution:
                    break

            except:
                raise Error
            if not status:
                lg.info("Break! Murty solver is over!")
                break
            lg.debug(f"murty step = {murty_step}, solution = {solution}")
            next_log_weight = self.global_hypothesis.log_weight - solution_cost
            next_associations = self.assignment_to_associations(solution)
            next_global_hypothesis = GlobalHypothesis(
                next_log_weight, next_associations
            )
            lg.debug(f"next global hypothesis: {next_global_hypothesis}")
            new_global_hypotheses.append(next_global_hypothesis)
        return new_global_hypotheses

    def assignment_to_associations(self, solution):
        associations = []
        for measurement_row, target_column in enumerate(solution):
            if target_column + 1 > self.num_of_old_tracks:
                # assignment is to new target
                track_id, sth_id = self.column_row_to_new_detected_sth[
                    target_column - self.num_of_old_tracks
                ]
            else:
                # assignment is to a previously detected target
                (
                    track_id,
                    parent_sth_id,
                    child_idx,
                ) = self.column_row_to_detected_child_sth[target_column][
                    measurement_row
                ]
                sth_id = (
                    self.old_tracks[track_id]
                    .single_target_hypotheses[parent_sth_id]
                    .detection_hypotheses[child_idx]
                    .sth_id
                )
            associations.append(Association(track_id, sth_id))
        return associations

    def create_cost_for_associated_targets(
        self, global_hypothesis: GlobalHypothesis, old_tracks, measurements
    ) -> np.ndarray:
        cost_detected = np.full(
            (len(measurements), len(global_hypothesis.associations)), np.inf
        )
        for column_idx, (track_idx, parent_sth_idx) in enumerate(
            global_hypothesis.associations
        ):
            parent_sth = old_tracks[track_idx].single_target_hypotheses[parent_sth_idx]
            for meas_idx, sth in parent_sth.detection_hypotheses.items():
                cost_detected[meas_idx, column_idx] = sth.cost
                self.column_row_to_detected_child_sth[column_idx][meas_idx] = (
                    track_idx,
                    parent_sth_idx,
                    meas_idx,
                )
        return cost_detected

    def create_cost_for_undetected(self, new_tracks, measurements) -> np.ndarray:
        # Using association between measurements and previously undetected objects

        cost_undetected = np.full((len(measurements), len(measurements)), np.inf)

        sth_idx = 0  # we have olny one sth for new targets
        for meas_idx in range(len(measurements)):
            if meas_idx in [
                track.single_target_hypotheses[sth_idx].meas_idx
                for track in new_tracks.values()
            ]:
                track_id = [
                    track.track_id
                    for track in new_tracks.values()
                    if track.single_target_hypotheses[sth_idx].meas_idx == meas_idx
                ][0]
                cost_undetected[meas_idx, meas_idx] = (
                    new_tracks[track_id].single_target_hypotheses[sth_idx].cost
                )
                self.column_row_to_new_detected_sth[meas_idx] = (
                    new_tracks[track_id].track_id,
                    sth_idx,
                )

        return cost_undetected
