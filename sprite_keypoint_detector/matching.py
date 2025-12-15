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
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """Find top N clothed frames by joint distance.

    Args:
        base_frame: Name of base frame
        base_keypoints: Keypoints for base frame
        all_clothed_annotations: All clothed frame annotations
        top_n: Number of candidates to return

    Returns:
        List of (clothed_frame_name, joint_distance) tuples, sorted by distance
    """
    distances = []

    for clothed_frame, clothed_data in all_clothed_annotations.items():
        clothed_kpts = clothed_data.get("keypoints", {})
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
    blue_threshold: int = 2000
) -> Tuple[MatchCandidate, bool]:
    """Select best candidate, minimizing blue with red as tiebreaker.

    Args:
        candidates: List of scored candidates
        blue_threshold: Flag for review if best blue > this

    Returns:
        (best_candidate, needs_review)
    """
    # Sort by blue (primary), then red (tiebreaker within 5%)
    sorted_candidates = sorted(candidates, key=lambda c: (c.blue_pixels, c.red_pixels))

    best = sorted_candidates[0]

    # Check for tiebreaker situation (within 5% blue)
    if len(sorted_candidates) > 1:
        second = sorted_candidates[1]
        if best.blue_pixels > 0:
            ratio = abs(best.blue_pixels - second.blue_pixels) / best.blue_pixels
            if ratio < 0.05 and second.red_pixels < best.red_pixels:
                best = second

    needs_review = best.blue_pixels > blue_threshold

    return best, needs_review
