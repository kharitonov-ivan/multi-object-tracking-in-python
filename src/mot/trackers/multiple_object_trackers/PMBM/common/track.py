import pprint

from .single_target_hypothesis import SingleTargetHypothesis


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
        sth_rep = (
            f"STH: \n {pprint.pformat(self.single_target_hypotheses)}"
            if len(self.single_target_hypotheses) < 3
            else ""
        )
        return (
            self.__class__.__name__
            + (
                f" id = {self.track_id}"
                f" number of sth = {len(self.single_target_hypotheses)} "
            )
            + sth_rep
        )

    @classmethod
    def from_sth(cls, single_target_hypo):
        return cls(initial_sth=single_target_hypo)

    def cut_tree(self):
        new_single_target_hypotheses = {}
        for parent_sth in self.single_target_hypotheses.values():
            for child_sth in parent_sth.detection_hypotheses.values():
                new_single_target_hypotheses[child_sth.sth_id] = child_sth
        self.single_target_hypotheses = new_single_target_hypotheses
