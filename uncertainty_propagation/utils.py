"""utils.py is almost always a code smell, but we accept it during development"""

import numpy as np
from scipy import stats


def append_or_assign(history: np.ndarray | None, addition: np.ndarray) -> np.ndarray:
    if history is None:
        return addition
    return np.append(history, addition, axis=0)


def extend_cache(
    history_x: np.ndarray | None,
    history_y: np.ndarray | None,
    new_x: np.ndarray,
    new_y: np.ndarray,
    cache_x: bool = False,
    cache_y: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cache_x:
        history_x = append_or_assign(history_x, new_x)
    if cache_y:
        history_y = append_or_assign(history_y, new_y)
    return history_x, history_y


def safety_index(probability: float):
    return -stats.norm.ppf(probability)
