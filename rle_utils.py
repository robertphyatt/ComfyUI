# rle_utils.py
"""Run-length encoding utilities for mask compression."""
import numpy as np
from typing import List, Dict

def encode_rle(mask: np.ndarray) -> List[Dict[str, int]]:
    """Encode 1D binary mask as run-length encoded list.

    Args:
        mask: 1D numpy array of 0s and 1s

    Returns:
        List of {"value": int, "count": int} dicts

    Example:
        [0, 0, 0, 1, 1, 0] -> [{"value": 0, "count": 3}, {"value": 1, "count": 2}, {"value": 0, "count": 1}]
    """
    if len(mask) == 0:
        return []

    rle = []
    current_value = int(mask[0])
    current_count = 1

    for i in range(1, len(mask)):
        if mask[i] == current_value:
            current_count += 1
        else:
            rle.append({"value": current_value, "count": current_count})
            current_value = int(mask[i])
            current_count = 1

    # Append final run
    rle.append({"value": current_value, "count": current_count})

    return rle

def decode_rle(rle: List[Dict[str, int]], length: int) -> np.ndarray:
    """Decode run-length encoded list to 1D binary mask.

    Args:
        rle: List of {"value": int, "count": int} dicts
        length: Expected length of output array

    Returns:
        1D numpy array of 0s and 1s
    """
    mask = np.zeros(length, dtype=np.uint8)
    pos = 0

    for run in rle:
        value = run["value"]
        count = run["count"]
        mask[pos:pos + count] = value
        pos += count

    return mask
