from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GlobalHypothesis:
    log_weight: float
    associations: Tuple[Tuple[int, int]]  # (num_of_tree, num_of_hypo_in_tree)

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(w={self.log_weight:.2f}, " f"(track_id, sth_id)={self.associations}, "
        )
