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


# Joint-specific thresholds for matching
# Arms/shoulders are critical for armor - strict threshold
# Ankles/toes are less important - looser threshold
JOINT_THRESHOLDS = {
    # Critical joints for armor matching (strict)
    'left_shoulder': 20.0,
    'right_shoulder': 20.0,
    'left_elbow': 20.0,
    'right_elbow': 20.0,
    'left_wrist': 25.0,
    'right_wrist': 25.0,
    'left_fingertip': 30.0,
    'right_fingertip': 30.0,
    # Core body (medium)
    'head': 25.0,
    'neck': 20.0,
    'left_hip': 25.0,
    'right_hip': 25.0,
    # Legs (medium-loose)
    'left_knee': 30.0,
    'right_knee': 30.0,
    # Feet (loose - less critical for armor)
    'left_ankle': 40.0,
    'right_ankle': 40.0,
    'left_toe': 50.0,
    'right_toe': 50.0,
}


def exceeds_per_joint_threshold(
    base_keypoints: Dict,
    clothed_keypoints: Dict,
    keypoint_names: List[str],
    max_per_joint: float = 80.0
) -> Tuple[bool, Optional[str], float]:
    """Check if any single joint exceeds its distance threshold.

    Uses joint-specific thresholds from JOINT_THRESHOLDS dict.
    Arms/shoulders are strict (20px) since they're critical for armor.
    Ankles/toes are looser (40-50px) since they matter less.
    Falls back to max_per_joint for unknown joints.

    Args:
        base_keypoints: Keypoints dict for base frame
        clothed_keypoints: Keypoints dict for clothed frame
        keypoint_names: List of keypoint names to compare
        max_per_joint: Default threshold for joints not in JOINT_THRESHOLDS

    Returns:
        (exceeds, worst_joint_name, worst_joint_distance)
        exceeds is True if any joint exceeds its threshold
    """
    per_joint = compute_per_joint_distances(base_keypoints, clothed_keypoints, keypoint_names)

    worst_joint = None
    worst_dist = 0.0
    exceeds = False

    for name, dist in per_joint.items():
        # Get joint-specific threshold or fall back to default
        threshold = JOINT_THRESHOLDS.get(name, max_per_joint)

        if dist == float('inf'):
            # Missing critical joints (strict threshold) should fail the check
            if threshold <= 25.0:
                return True, name, float('inf')
            # Allow missing loose joints (ankles, toes)
            continue

        if dist > threshold:
            exceeds = True
            # Track the worst violation by absolute distance
            if worst_joint is None or dist > worst_dist:
                worst_dist = dist
                worst_joint = name

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
            # Note: Uncomment for debugging threshold issues
            # print(f"    Filtering {clothed_frame}: {worst_joint} exceeds threshold ({worst_dist:.1f}px > {max_per_joint}px)")
            continue

        dist = compute_joint_distance(base_keypoints, clothed_kpts, KEYPOINT_NAMES)
        distances.append((clothed_frame, dist))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    if not distances:
        print(f"  WARNING: All candidates filtered by per-joint threshold ({max_per_joint}px)")

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
