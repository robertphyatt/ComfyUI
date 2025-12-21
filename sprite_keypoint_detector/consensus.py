"""Color consensus correction using majority voting across frames.

Maps pixels to canonical segment-relative coordinates, votes on the correct
palette index for each position, and corrects outliers to match consensus.
"""

import numpy as np
from typing import Dict, List, Tuple
from .consistency import BONE_SEGMENTS, find_palette_index


def pixel_to_canonical(
    pixel_pos: Tuple[int, int],
    joint_a: np.ndarray,
    joint_b: np.ndarray
) -> Tuple[float, float]:
    """Convert pixel position to canonical segment coordinates.

    Canonical coords: x=0 at joint_a, x=1 at joint_b, y=perpendicular distance
    normalized by segment length.

    Args:
        pixel_pos: (x, y) pixel position
        joint_a: Segment start joint position
        joint_b: Segment end joint position

    Returns:
        (canonical_x, canonical_y) normalized coordinates
    """
    seg_vec = joint_b - joint_a
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1:
        return (0.0, 0.0)

    seg_unit = seg_vec / seg_len
    perp = np.array([-seg_unit[1], seg_unit[0]])

    to_pixel = np.array(pixel_pos, dtype=float) - joint_a
    along = np.dot(to_pixel, seg_unit) / seg_len
    across = np.dot(to_pixel, perp) / seg_len

    return (float(along), float(across))


def discretize_canonical(
    x: float,
    y: float,
    resolution: float = 0.05
) -> Tuple[int, int]:
    """Round canonical coords to discrete grid position.

    Args:
        x: Canonical x coordinate
        y: Canonical y coordinate
        resolution: Grid resolution (default 0.05 = 20 bins per unit)

    Returns:
        (grid_x, grid_y) discrete grid position
    """
    return (int(round(x / resolution)), int(round(y / resolution)))
