from typing import List


class SingleTargetHypothesis:
    def __init__(
        self, bernoulli, likelihood: float, associated_measurement_idx: int, cost
    ):
        self.bernoulli = bernoulli
        self.likelihood = likelihood
        self.cost = cost
        self.associated_measurement_idx = (
            associated_measurement_idx  # if -1 - misdetection
        )
        self.missdetection_hypothesis = None
        self.detection_hypotheses = {}

    def get_child_by_meas_idx(self, meas_idx: int):
        child_by_meas_idx = {
            v.associated_measurement_idx: k for (k, v) in self.detection_hypotheses.items()
        }
        if meas_idx in child_by_meas_idx.keys():
            sth_id = child_by_meas_idx[meas_idx]
            return self.detection_hypotheses[sth_id]
        else:
            raise KeyError

    def get_child_sth_idx_by_meas_idx(self, meas_idx: int):
        child_by_meas_idx = {
            v.associated_measurement_idx: k for (k, v) in self.detection_hypotheses.items()
        }
        if meas_idx in child_by_meas_idx.keys():
            sth_id = child_by_meas_idx[meas_idx]
            return sth_id
        else:
            raise KeyError

    def __repr__(self):
        return (
            f"likelihood={self.likelihood:.2f}, "
            f"meas_idx={self.associated_measurement_idx}, "
        )


class Track:
    """Represents a track - hypotheses tree.
    The root is association with unique target.
    Leafs are hypotheses which represent association of this target
    with corresponding measurements or missdetections.
    """

    max_track_id = 1000000
    track_id_generator = (x for x in range(max_track_id))

    def __init__(self, initial_sth=None):
        self.track_id = next(Track.track_id_generator)
        self.max_sth_id = 1000000
        self.sth_id_generator = (x for x in range(self.max_sth_id))
        self.single_target_hypotheses = {next(self.sth_id_generator): initial_sth}

    def add_sth(self, sth: SingleTargetHypothesis) -> None:
        self.single_target_hypotheses[next(self.sth_id_generator)] = sth

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f" id = {self.track_id}"
            f" number of sth = {len(self.single_target_hypotheses}"
        )

    @classmethod
    def from_sth(cls, single_target_hypo):
        return cls(initial_sth=single_target_hypo)

    def cut_tree(self):
        new_single_target_hypotheses = {}
        for parent_sth_idx, parent_sth in self.single_target_hypotheses.items():
            for child_sth_idx, child_sth in parent_sth.children.items():
                new_single_target_hypotheses[child_sth_idx] = child_sth
        self.single_target_hypotheses = new_single_target_hypotheses
