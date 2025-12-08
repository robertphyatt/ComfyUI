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
    # Validate input is 1D
    if mask.ndim != 1:
        raise ValueError(f"Input mask must be 1D array, got shape {mask.shape}")

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
    # Validate required keys and values
    total_count = 0
    for run in rle:
        # Validate required keys exist
        if "value" not in run or "count" not in run:
            raise ValueError(f"RLE entry missing required keys: {run}")

        value = run["value"]
        count = run["count"]

        # Validate values are 0 or 1 only
        if value not in (0, 1):
            raise ValueError(f"RLE values must be 0 or 1, got {value}")

        # Validate no negative counts
        if count < 0:
            raise ValueError(f"RLE count must be non-negative, got {count}")

        total_count += count

    # Validate RLE sum matches expected length
    if total_count != length:
        raise ValueError(f"RLE sum ({total_count}) does not match expected length ({length})")

    mask = np.zeros(length, dtype=np.uint8)
    pos = 0

    for run in rle:
        value = run["value"]
        count = run["count"]
        mask[pos:pos + count] = value
        pos += count

    return mask
