import numpy as np


# Legacy simulator right-hand order:
# [thumb0, thumb1, thumb2, middle0, middle1, index0, index1]
# Policy/Dataset right-hand order:
# [thumb0, thumb1, thumb2, index0, index1, middle0, middle1]
#
# This permutation is self-inverse, so it can be used in both directions.
DEX3_RIGHT_LEGACY_SIM_PERM = np.array([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)


def reorder_dex3_right_legacy_sim(values, *, context: str = "") -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[-1] != 7:
        where = f" ({context})" if context else ""
        raise ValueError(f"Dex3 right-hand reorder expects last dim 7{where}, got shape={arr.shape}.")
    return arr[..., DEX3_RIGHT_LEGACY_SIM_PERM]
