from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Association:
    track_id: int # num of tree
    sth_id: int # num of hypo in tree

    def __iter__(self):
        return iter((self.track_id, self.sth_id))

@dataclass
class GlobalHypothesis:
    log_weight: float
    associations: Tuple[Association]

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(w={self.log_weight:.2f}, " f"(track_id, sth_id)={self.associations}, "
        )
