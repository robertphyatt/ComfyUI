"""Color consensus correction using majority voting across frames.

Maps pixels to canonical segment-relative coordinates, votes on the correct
palette index for each position, and corrects outliers to match consensus.
"""

import numpy as np
from collections import Counter
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


def get_segment_pixels_with_positions(
    frame: np.ndarray,
    keypoints: np.ndarray,
    joint_a_idx: int,
    joint_b_idx: int,
    segment_width: int = 50
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Get visible pixels in segment with their canonical grid positions.

    Args:
        frame: RGBA frame
        keypoints: Keypoint array (14 joints)
        joint_a_idx: Index of joint A
        joint_b_idx: Index of joint B
        segment_width: Width of segment region in pixels

    Returns:
        List of ((pixel_x, pixel_y), (grid_x, grid_y)) tuples
    """
    h, w = frame.shape[:2]
    joint_a = keypoints[joint_a_idx]
    joint_b = keypoints[joint_b_idx]

    seg_vec = joint_b - joint_a
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1:
        return []

    seg_unit = seg_vec / seg_len
    perp = np.array([-seg_unit[1], seg_unit[0]])

    results = []
    half_width = segment_width / 2

    for y in range(h):
        for x in range(w):
            if frame[y, x, 3] <= 128:
                continue

            to_pixel = np.array([x, y], dtype=float) - joint_a
            along = np.dot(to_pixel, seg_unit)
            across = np.dot(to_pixel, perp)

            if along < 0 or along > seg_len or abs(across) > half_width:
                continue

            cx = along / seg_len
            cy = across / seg_len
            gx, gy = discretize_canonical(cx, cy)

            results.append(((x, y), (gx, gy)))

    return results


def build_consensus_map(
    frames: List[np.ndarray],
    keypoints_per_frame: List[np.ndarray],
    palette: np.ndarray,
    segment_width: int = 50
) -> Dict[Tuple[int, int, int], int]:
    """Build consensus palette index for each segment grid position.

    Collects votes from all frames, returns plurality winner for each position.

    Args:
        frames: List of RGBA frames
        keypoints_per_frame: Keypoints for each frame
        palette: Color palette (n_colors, 3) BGR
        segment_width: Width for segment regions

    Returns:
        Dict mapping (segment_idx, grid_x, grid_y) -> consensus palette index
    """
    votes: Dict[Tuple[int, int, int], List[int]] = {}

    for frame, keypoints in zip(frames, keypoints_per_frame):
        for seg_idx, (joint_a_idx, joint_b_idx, name) in enumerate(BONE_SEGMENTS):
            pixels_with_positions = get_segment_pixels_with_positions(
                frame, keypoints, joint_a_idx, joint_b_idx, segment_width
            )

            for (px, py), (gx, gy) in pixels_with_positions:
                key = (seg_idx, gx, gy)
                color = frame[py, px, :3]
                palette_idx = find_palette_index(color, palette)

                if key not in votes:
                    votes[key] = []
                votes[key].append(palette_idx)

    consensus_map = {}
    for key, idx_list in votes.items():
        counter = Counter(idx_list)
        winner = counter.most_common(1)[0][0]
        consensus_map[key] = winner

    return consensus_map
