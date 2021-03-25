from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GlobalHypothesis:
    weight: float
    hypothesis: Tuple[Tuple[int, int]]  # (num_of_tree, num_of_hypo_in_tree)

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(w={self.weight:.2f}, " f"(track_id, sth_id)={self.hypothesis}, "
        )
