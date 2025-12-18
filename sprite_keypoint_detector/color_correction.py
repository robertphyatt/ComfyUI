"""Color correction using golden frame reference.

Maps pixel colors from a golden reference frame to all other frames
using body-segment-based keypoint-relative positioning.
"""

import numpy as np
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class BodySegment(IntEnum):
    """Body segment identifiers."""
    HEAD = 0
    TORSO = 1
    LEFT_UPPER_ARM = 2
    LEFT_LOWER_ARM = 3
    RIGHT_UPPER_ARM = 4
    RIGHT_LOWER_ARM = 5
    LEFT_UPPER_LEG = 6
    LEFT_LOWER_LEG = 7
    RIGHT_UPPER_LEG = 8
    RIGHT_LOWER_LEG = 9


# Keypoint indices for each segment (start, end)
# Based on keypoints.py: 0=head, 1=neck, 2=left_shoulder, 3=right_shoulder,
# 4=left_elbow, 5=right_elbow, 6=left_wrist, 7=right_wrist,
# 8=left_fingertip, 9=right_fingertip, 10=left_hip, 11=right_hip,
# 12=left_knee, 13=right_knee, 14=left_ankle, 15=right_ankle,
# 16=left_toe, 17=right_toe
SEGMENT_KEYPOINTS: Dict[BodySegment, Tuple[int, int]] = {
    BodySegment.HEAD: (0, 1),           # head -> neck
    BodySegment.TORSO: (1, 10),         # neck -> left_hip (use as torso anchor)
    BodySegment.LEFT_UPPER_ARM: (2, 4),  # left_shoulder -> left_elbow
    BodySegment.LEFT_LOWER_ARM: (4, 6),  # left_elbow -> left_wrist
    BodySegment.RIGHT_UPPER_ARM: (3, 5), # right_shoulder -> right_elbow
    BodySegment.RIGHT_LOWER_ARM: (5, 7), # right_elbow -> right_wrist
    BodySegment.LEFT_UPPER_LEG: (10, 12), # left_hip -> left_knee
    BodySegment.LEFT_LOWER_LEG: (12, 14), # left_knee -> left_ankle
    BodySegment.RIGHT_UPPER_LEG: (11, 13), # right_hip -> right_knee
    BodySegment.RIGHT_LOWER_LEG: (13, 15), # right_knee -> right_ankle
}


@dataclass
class PixelPosition:
    """Relative position of a pixel within a body segment."""
    segment: BodySegment
    along_bone: float      # 0.0 = at start keypoint, 1.0 = at end keypoint
    perpendicular: float   # signed distance perpendicular to bone (pixels)


def _point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> Tuple[float, float]:
    """Compute distance from point to line segment.

    Returns:
        (along_bone, perpendicular): Position along bone (0-1) and signed perpendicular distance
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        # Degenerate line segment
        return 0.5, np.linalg.norm(point - line_start)

    line_unit = line_vec / line_len
    point_vec = point - line_start

    # Project point onto line
    along = np.dot(point_vec, line_unit) / line_len  # 0-1 range (can be outside)

    # Perpendicular distance (signed: positive = left of bone direction)
    perp_vec = point_vec - (along * line_len) * line_unit
    perp_dist = np.linalg.norm(perp_vec)

    # Sign: use cross product to determine which side
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    if cross < 0:
        perp_dist = -perp_dist

    return along, perp_dist


def assign_pixel_to_segment(
    pixel_y: int,
    pixel_x: int,
    keypoints: np.ndarray
) -> Optional[PixelPosition]:
    """Assign a pixel to its nearest body segment.

    Args:
        pixel_y: Pixel Y coordinate
        pixel_x: Pixel X coordinate
        keypoints: 18x2 array of keypoint coordinates

    Returns:
        PixelPosition with segment and relative position, or None if too far from any segment
    """
    point = np.array([pixel_x, pixel_y], dtype=np.float64)

    best_segment = None
    best_distance = float('inf')
    best_along = 0.0
    best_perp = 0.0

    for segment, (kp_start, kp_end) in SEGMENT_KEYPOINTS.items():
        start = keypoints[kp_start]
        end = keypoints[kp_end]

        along, perp = _point_to_line_distance(point, start, end)

        # Distance to segment (clamped along to 0-1 for distance calc)
        clamped_along = max(0.0, min(1.0, along))
        closest_on_segment = start + clamped_along * (end - start)
        distance = np.linalg.norm(point - closest_on_segment)

        if distance < best_distance:
            best_distance = distance
            best_segment = segment
            best_along = along
            best_perp = perp

    if best_segment is None:
        return None

    return PixelPosition(
        segment=best_segment,
        along_bone=best_along,
        perpendicular=best_perp
    )
