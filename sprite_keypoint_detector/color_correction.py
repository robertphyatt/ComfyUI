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
