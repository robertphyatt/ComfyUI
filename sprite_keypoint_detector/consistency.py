"""Frame-to-frame pixel consistency checker."""

import numpy as np
from typing import Tuple, List
import cv2
from math import atan2, cos, sin


# Bone segments: (joint_a_idx, joint_b_idx, name)
BONE_SEGMENTS = [
    (1, 0, "head_neck"),
    (2, 4, "l_upper_arm"),
    (4, 6, "l_forearm"),
    (3, 5, "r_upper_arm"),
    (5, 7, "r_forearm"),
    (8, 10, "l_thigh"),
    (10, 12, "l_shin"),
    (9, 11, "r_thigh"),
    (11, 13, "r_shin"),
]


def compute_segment_transform(
    joint_a_n: np.ndarray,
    joint_b_n: np.ndarray,
    joint_a_n1: np.ndarray,
    joint_b_n1: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute translation and rotation between frame N and N+1 for a segment.

    Args:
        joint_a_n: Joint A position in frame N (x, y)
        joint_b_n: Joint B position in frame N (x, y)
        joint_a_n1: Joint A position in frame N+1 (x, y)
        joint_b_n1: Joint B position in frame N+1 (x, y)

    Returns:
        (translation, rotation_angle, center_n) where:
        - translation: (dx, dy) movement of segment midpoint
        - rotation_angle: angle change in radians
        - center_n: midpoint in frame N (rotation center)
    """
    # Midpoints
    mid_n = (joint_a_n + joint_b_n) / 2
    mid_n1 = (joint_a_n1 + joint_b_n1) / 2

    # Translation
    translation = mid_n1 - mid_n

    # Rotation angles
    angle_n = atan2(joint_b_n[1] - joint_a_n[1], joint_b_n[0] - joint_a_n[0])
    angle_n1 = atan2(joint_b_n1[1] - joint_a_n1[1], joint_b_n1[0] - joint_a_n1[0])
    rotation = angle_n1 - angle_n

    return translation, rotation, mid_n


def warp_pixel_position(
    pixel_pos: Tuple[int, int],
    center: np.ndarray,
    translation: np.ndarray,
    rotation: float
) -> Tuple[int, int]:
    """Warp a pixel position from frame N to expected position in frame N+1.

    Args:
        pixel_pos: (x, y) position in frame N
        center: rotation center (segment midpoint in frame N)
        translation: (dx, dy) translation
        rotation: rotation angle in radians

    Returns:
        (x, y) expected position in frame N+1
    """
    # Translate to origin (center)
    px, py = pixel_pos[0] - center[0], pixel_pos[1] - center[1]

    # Rotate
    cos_r, sin_r = cos(rotation), sin(rotation)
    rx = px * cos_r - py * sin_r
    ry = px * sin_r + py * cos_r

    # Translate back and apply movement
    new_x = rx + center[0] + translation[0]
    new_y = ry + center[1] + translation[1]

    return int(round(new_x)), int(round(new_y))
