import itertools
import logging as lg
from collections import defaultdict
from typing import Dict, List

import numpy as np
from murty import Murty

from .global_hypothesis import Association, GlobalHypothesis


def gibbs_sampling(cost_matrix: np.ndarray, num_iterations: int):
    num_elements = cost_matrix.shape[0]
    num_assignments = cost_matrix.shape[1]

    # Инициализируем произвольное назначение
    assignment = np.random.randint(0, num_assignments, size=num_elements)
    cost_current = np.sum(cost_matrix[np.arange(num_elements), assignment])

    costs = np.empty(num_iterations)

    for i in range(num_iterations):
        # Выбираем случайный элемент для переназначения
        element = np.random.choice(num_elements)

        # Генерируем новые назначения и вычисляем их затраты
        new_assignment = assignment.copy()
        new_assignment[element] = np.random.randint(0, num_assignments)
        new_cost = np.sum(cost_matrix[np.arange(num_elements), new_assignment])

        # Принимаем новое назначение, если оно улучшает затраты
        if new_cost < cost_current:
            assignment = new_assignment
            cost_current = new_cost

        costs[i] = cost_current

    return assignment, costs


class CostMatrix:
    def __init__(
        self,
        global_hypothesis: GlobalHypothesis,
        old_tracks: Dict,
        new_tracks: Dict,
        measurements,
    ) -> None:
        self.global_hypothesis = global_hypothesis
        self.old_tracks = old_tracks
        self.new_tracks = new_tracks
        self.measurements = measurements
        self.column_row_to_detected_child_sth = defaultdict(defaultdict)
        self.column_row_to_new_detected_sth = defaultdict(defaultdict)
        self.num_of_old_tracks = len(list(self.global_hypothesis.associations))
        self.cost_matrix = self.create_cost_matrix()

    def __repr__(self) -> str:
        return f"cost matrix = {self.cost_matrix}"

    def create_cost_matrix(self):
        cost_detected = self.create_cost_for_associated_targets(
            self.global_hypothesis, self.old_tracks, self.measurements
        )
        cost_undetected = self.create_cost_for_undetected(
            self.new_tracks, self.measurements
        )
        cost_matrix = np.hstack([cost_detected, cost_undetected])
        return cost_matrix

    def create_cost_for_associated_targets(
        self, global_hypothesis: GlobalHypothesis, old_tracks, measurements
    ) -> np.ndarray:
        cost_detected = np.full(
            (len(measurements), len(list(global_hypothesis.associations))), np.inf
        )
        for column_idx, (track_idx, parent_sth_idx) in enumerate(
            global_hypothesis.associations
        ):
            parent_sth = old_tracks[track_idx].single_target_hypotheses[parent_sth_idx]
            for meas_idx, sth in parent_sth.detection_hypotheses.items():
                cost_detected[:, column_idx] = sth.cost
                self.column_row_to_detected_child_sth[column_idx][meas_idx] = (
                    track_idx,
                    parent_sth_idx,
                    meas_idx,
                    sth.sth_id,
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
                self.column_row_to_new_detected_sth[meas_idx] = Association(
                    new_tracks[track_id].track_id,
                    sth_idx,
                )

        return cost_undetected

    def optimized_assignment_to_associations(self, solution):
        new_target_rows = np.argwhere(solution + 1 > self.num_of_old_tracks)
        new_target_columns = solution[new_target_rows] - self.num_of_old_tracks
        new_associations = (
            self.column_row_to_new_detected_sth[target_column.item()]
            for target_column in new_target_columns
        )

        previous_target_rows = np.argwhere(solution + 1 < self.num_of_old_tracks)
        previous_target_columns = solution[previous_target_rows]

        gen1 = (
            self.column_row_to_detected_child_sth[target_column.item()][
                target_row.item()
            ]
            for (target_row, target_column) in zip(
                previous_target_rows, previous_target_columns
            )
        )
        previous_target_associations = (
            Association(track_id, sth_id)
            for (track_id, parent_sth_id, child_idx, sth_id) in gen1
        )
        result = itertools.chain(new_associations, previous_target_associations)
        return list(result)

    def assignment_to_associations(self, solution):
        associations = []
        for measurement_row, target_column in np.ndenumerate(solution):
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
                    _,
                ) = self.column_row_to_detected_child_sth[target_column][
                    measurement_row[0]
                ]
                sth_id = (
                    self.old_tracks[track_id]
                    .single_target_hypotheses[parent_sth_id]
                    .detection_hypotheses[child_idx]
                    .sth_id
                )
            associations.append(Association(track_id, sth_id))
        return associations


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
        self.global_hypothesis = global_hypothesis
        self.cost_matrix = CostMatrix(
            global_hypothesis, old_tracks, new_tracks, measurements
        )
        self.num_of_desired_hypotheses = num_of_desired_hypotheses
        self.max_murty_steps = max_murty_steps or self.get_murty_steps()

        if self.cost_matrix.cost_matrix.size == 0:
            return

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"cost_matrix={self.cost_matrix}, )")

    def get_murty_steps(self):
        # TODO add docstring
        return int(
            np.ceil(
                np.exp(self.global_hypothesis.log_weight)
                * self.num_of_desired_hypotheses
            )
        )

    def solve(self) -> List[GlobalHypothesis]:
        global_hypo_list = []
        # a, c = assign_2d_by_gibbs(self.cost_matrix.cost_matrix, 10, self.max_murty_steps)

        # for i in range(self.max_murty_steps):
        #     if c[i] == -1:
        #         break
        #     global_hypo_list.append(GlobalHypothesis(log_weight=self.global_hypothesis.log_weight - c[i], associations= self.cost_matrix.assignment_to_associations(a[i])) )

        murty_solver = Murty(self.cost_matrix.cost_matrix)

        for _ in range(self.max_murty_steps):
            status, solution_cost, murty_solution = murty_solver.draw()

            if not status:
                lg.debug("Murty was broken")
                break
            else:
                current_log_weight = self.global_hypothesis.log_weight - solution_cost
                current_association = self.cost_matrix.assignment_to_associations(
                    murty_solution
                )
                global_hypo = GlobalHypothesis(
                    log_weight=current_log_weight, associations=current_association
                )
                global_hypo_list.append(global_hypo)

        return global_hypo_list


def assign(
    global_hypothesis,
    old_tracks,
    new_tracks,
    measurements,
    num_of_desired_hypotheses,
    max_murty_steps=None,
):
    problem = AssignmentSolver(
        global_hypothesis,
        old_tracks,
        new_tracks,
        measurements,
        num_of_desired_hypotheses,
        max_murty_steps=None,
    )
    return problem.solve()
