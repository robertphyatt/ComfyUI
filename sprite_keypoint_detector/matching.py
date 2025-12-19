"""Frame matching: find best clothed frame for each base frame."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2

from .keypoints import KEYPOINT_NAMES


@dataclass
class MatchCandidate:
    """A candidate clothed frame match for a base frame."""
    clothed_frame: str
    joint_distance: float
    blue_pixels: int  # After alignment + rotation
    red_pixels: int
    score_rank: int


@dataclass
class FrameMatch:
    """Final match result for a base frame."""
    base_frame: str
    matched_clothed_frame: str
    candidates: List[MatchCandidate]
    needs_review: bool  # True if best blue > threshold


def compute_joint_distance(
    base_keypoints: Dict,
    clothed_keypoints: Dict,
    keypoint_names: List[str]
) -> float:
    """Compute total euclidean distance between corresponding joints.

    Args:
        base_keypoints: Keypoints dict for base frame
        clothed_keypoints: Keypoints dict for clothed frame
        keypoint_names: List of keypoint names to compare

    Returns:
        Sum of euclidean distances between all valid keypoint pairs
    """
    total_dist = 0.0
    valid_pairs = 0

    for name in keypoint_names:
        base_pt = _get_xy(base_keypoints, name)
        clothed_pt = _get_xy(clothed_keypoints, name)

        if base_pt and clothed_pt:
            dist = np.sqrt((base_pt[0] - clothed_pt[0])**2 +
                          (base_pt[1] - clothed_pt[1])**2)
            total_dist += dist
            valid_pairs += 1

    # Penalize missing keypoints
    if valid_pairs < len(keypoint_names):
        missing_penalty = (len(keypoint_names) - valid_pairs) * 100
        total_dist += missing_penalty

    return total_dist


def compute_per_joint_distances(
    base_keypoints: Dict,
    clothed_keypoints: Dict,
    keypoint_names: List[str]
) -> Dict[str, float]:
    """Compute euclidean distance for each joint individually.

    Args:
        base_keypoints: Keypoints dict for base frame
        clothed_keypoints: Keypoints dict for clothed frame
        keypoint_names: List of keypoint names to compare

    Returns:
        Dict mapping joint name to distance (missing joints get infinity)
    """
    distances = {}

    for name in keypoint_names:
        base_pt = _get_xy(base_keypoints, name)
        clothed_pt = _get_xy(clothed_keypoints, name)

        if base_pt and clothed_pt:
            dist = np.sqrt((base_pt[0] - clothed_pt[0])**2 +
                          (base_pt[1] - clothed_pt[1])**2)
            distances[name] = dist
        else:
            distances[name] = float('inf')

    return distances


def exceeds_per_joint_threshold(
    base_keypoints: Dict,
    clothed_keypoints: Dict,
    keypoint_names: List[str],
    max_per_joint: float = 80.0
) -> Tuple[bool, Optional[str], float]:
    """Check if any single joint exceeds the per-joint distance threshold.

    This catches cases where total distance is acceptable but one limb
    (e.g., arm) is in a completely different position.

    Args:
        base_keypoints: Keypoints dict for base frame
        clothed_keypoints: Keypoints dict for clothed frame
        keypoint_names: List of keypoint names to compare
        max_per_joint: Maximum allowed distance for any single joint

    Returns:
        (exceeds, worst_joint_name, worst_joint_distance)
        exceeds is True if any joint exceeds threshold
    """
    per_joint = compute_per_joint_distances(base_keypoints, clothed_keypoints, keypoint_names)

    worst_joint = None
    worst_dist = 0.0

    for name, dist in per_joint.items():
        if dist > worst_dist and dist != float('inf'):
            worst_dist = dist
            worst_joint = name

    exceeds = worst_dist > max_per_joint

    return exceeds, worst_joint, worst_dist


def _get_xy(keypoints: Dict, name: str) -> Optional[Tuple[int, int]]:
    """Extract x,y from keypoint, handling both formats."""
    if name not in keypoints:
        return None
    kp = keypoints[name]
    if isinstance(kp, list):
        return (kp[0], kp[1])
    elif isinstance(kp, dict):
        return (kp["x"], kp["y"])
    return None


def find_top_candidates(
    base_frame: str,
    base_keypoints: Dict,
    all_clothed_annotations: Dict[str, Dict],
    top_n: int = 3,
    max_per_joint: float = 80.0
) -> List[Tuple[str, float]]:
    """Find top N clothed frames by joint distance.

    Filters out candidates where any single joint exceeds max_per_joint threshold.
    This prevents matching frames where one limb is in a completely different position.

    Args:
        base_frame: Name of base frame
        base_keypoints: Keypoints for base frame
        all_clothed_annotations: All clothed frame annotations
        top_n: Number of candidates to return
        max_per_joint: Max distance allowed for any single joint (default 80px)

    Returns:
        List of (clothed_frame_name, joint_distance) tuples, sorted by distance
    """
    distances = []

    for clothed_frame, clothed_data in all_clothed_annotations.items():
        clothed_kpts = clothed_data.get("keypoints", {})

        # Check per-joint threshold first
        exceeds, worst_joint, worst_dist = exceeds_per_joint_threshold(
            base_keypoints, clothed_kpts, KEYPOINT_NAMES, max_per_joint
        )

        if exceeds:
            # Skip this candidate - a joint is too far off
            continue

        dist = compute_joint_distance(base_keypoints, clothed_kpts, KEYPOINT_NAMES)
        distances.append((clothed_frame, dist))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    return distances[:top_n]


def score_candidate_after_transform(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int
) -> Tuple[int, int]:
    """Count blue (uncovered) and red (floating) pixels after transform.

    Args:
        base_image: Base frame RGBA
        armor_image: Transformed armor RGBA
        neck_y: Y coordinate of neck (only count below this)

    Returns:
        (blue_count, red_count)
    """
    h, w = base_image.shape[:2]

    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128

    # Only count below neck
    valid = np.zeros((h, w), dtype=bool)
    valid[neck_y:, :] = True

    blue = base_visible & ~armor_visible & valid  # Base uncovered
    red = armor_visible & ~base_visible & valid   # Armor floating

    return int(np.sum(blue)), int(np.sum(red))


def select_best_match(
    candidates: List[MatchCandidate],
    red_threshold: int = 2000
) -> Tuple[MatchCandidate, bool]:
    """Select best candidate by minimizing red pixels (floating armor).

    Blue pixels (uncovered base) are filled by inpainting and don't cause
    visual artifacts. Red pixels (armor extending beyond base silhouette)
    create visible halos and bloated limbs that can't be fixed.

    Therefore: minimize red first, use blue as tiebreaker.

    Args:
        candidates: List of scored candidates
        red_threshold: Flag for review if best red > this

    Returns:
        (best_candidate, needs_review)
    """
    # Sort by red (primary), then blue (tiebreaker)
    sorted_candidates = sorted(candidates, key=lambda c: (c.red_pixels, c.blue_pixels))

    best = sorted_candidates[0]
    needs_review = best.red_pixels > red_threshold

    return best, needs_review
