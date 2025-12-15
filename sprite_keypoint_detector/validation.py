"""Annotation validation with geometric sanity and confidence checks."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .keypoints import KEYPOINT_NAMES, SKELETON_CONNECTIONS


@dataclass
class ValidationResult:
    """Result of validating a single frame's annotations."""
    frame_name: str
    is_valid: bool
    issues: List[str]
    low_confidence_keypoints: List[str]


def get_keypoint_xy(keypoints: Dict, name: str) -> Optional[Tuple[int, int]]:
    """Extract x,y from keypoint, handling both legacy and new formats."""
    if name not in keypoints:
        return None
    kp = keypoints[name]
    if isinstance(kp, list):
        return (kp[0], kp[1])
    elif isinstance(kp, dict):
        return (kp["x"], kp["y"])
    return None


def get_confidence(keypoints: Dict, name: str) -> float:
    """Get confidence for a keypoint (1.0 for legacy/manual)."""
    if name not in keypoints:
        return 0.0
    kp = keypoints[name]
    if isinstance(kp, dict):
        return kp.get("confidence", 1.0)
    return 1.0  # Legacy format assumes high confidence


def compute_bone_lengths(keypoints: Dict) -> Dict[Tuple[int, int], float]:
    """Compute length of each skeleton bone."""
    lengths = {}
    for i, j in SKELETON_CONNECTIONS:
        name_i, name_j = KEYPOINT_NAMES[i], KEYPOINT_NAMES[j]
        pt_i = get_keypoint_xy(keypoints, name_i)
        pt_j = get_keypoint_xy(keypoints, name_j)
        if pt_i and pt_j:
            dist = np.sqrt((pt_i[0] - pt_j[0])**2 + (pt_i[1] - pt_j[1])**2)
            lengths[(i, j)] = dist
    return lengths


def validate_frame(
    frame_name: str,
    keypoints: Dict,
    image_bounds: Tuple[int, int] = (512, 512),
    confidence_threshold: float = 0.7,
    median_bone_lengths: Optional[Dict] = None
) -> ValidationResult:
    """Validate a single frame's annotations.

    Checks:
    1. Joints outside image bounds
    2. Limbs crossing (left wrist right of right wrist, etc.)
    3. Bone lengths wildly different from median (>2x or <0.5x)
    4. Low confidence keypoints

    Args:
        frame_name: Name of the frame being validated
        keypoints: Keypoint dict for this frame
        image_bounds: (width, height) of image
        confidence_threshold: Flag keypoints below this confidence
        median_bone_lengths: Optional median lengths for comparison

    Returns:
        ValidationResult with issues found
    """
    issues = []
    low_confidence = []
    w, h = image_bounds

    # Check 1: Joints outside bounds
    for name in KEYPOINT_NAMES:
        pt = get_keypoint_xy(keypoints, name)
        if pt:
            x, y = pt
            if x < 0 or x >= w or y < 0 or y >= h:
                issues.append(f"{name} outside bounds: ({x}, {y})")

    # Check 2: Limbs crossing (simple left/right checks)
    crossing_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]
    for left_name, right_name in crossing_pairs:
        left_pt = get_keypoint_xy(keypoints, left_name)
        right_pt = get_keypoint_xy(keypoints, right_name)
        if left_pt and right_pt:
            # Left should generally be to the left (lower x) of right
            # Allow some tolerance for crossed-arm poses
            if left_pt[0] > right_pt[0] + 50:  # 50px tolerance
                issues.append(f"Limbs may be crossing: {left_name} x={left_pt[0]} > {right_name} x={right_pt[0]}")

    # Check 3: Bone lengths vs median
    if median_bone_lengths:
        bone_lengths = compute_bone_lengths(keypoints)
        for (i, j), length in bone_lengths.items():
            if (i, j) in median_bone_lengths:
                median = median_bone_lengths[(i, j)]
                if median > 0:
                    ratio = length / median
                    if ratio > 2.0 or ratio < 0.5:
                        name_i, name_j = KEYPOINT_NAMES[i], KEYPOINT_NAMES[j]
                        issues.append(f"Bone {name_i}->{name_j} length {length:.1f} is {ratio:.1f}x median {median:.1f}")

    # Check 4: Low confidence
    for name in KEYPOINT_NAMES:
        conf = get_confidence(keypoints, name)
        if conf < confidence_threshold:
            low_confidence.append(f"{name} (conf={conf:.2f})")

    is_valid = len(issues) == 0 and len(low_confidence) == 0

    return ValidationResult(
        frame_name=frame_name,
        is_valid=is_valid,
        issues=issues,
        low_confidence_keypoints=low_confidence
    )


def compute_median_bone_lengths(all_keypoints: Dict[str, Dict]) -> Dict[Tuple[int, int], float]:
    """Compute median bone length across all frames."""
    all_lengths: Dict[Tuple[int, int], List[float]] = {}

    for frame_name, frame_data in all_keypoints.items():
        keypoints = frame_data.get("keypoints", {})
        lengths = compute_bone_lengths(keypoints)
        for bone, length in lengths.items():
            if bone not in all_lengths:
                all_lengths[bone] = []
            all_lengths[bone].append(length)

    medians = {}
    for bone, lengths in all_lengths.items():
        if lengths:
            medians[bone] = float(np.median(lengths))

    return medians


def validate_all_annotations(
    annotations: Dict[str, Dict],
    image_bounds: Tuple[int, int] = (512, 512),
    confidence_threshold: float = 0.7
) -> List[ValidationResult]:
    """Validate all frame annotations.

    Returns list of ValidationResults, with flagged frames first.
    """
    # Compute median bone lengths for comparison
    median_lengths = compute_median_bone_lengths(annotations)

    results = []
    for frame_name, frame_data in annotations.items():
        keypoints = frame_data.get("keypoints", {})
        result = validate_frame(
            frame_name, keypoints, image_bounds,
            confidence_threshold, median_lengths
        )
        results.append(result)

    # Sort: invalid first, then by number of issues
    results.sort(key=lambda r: (r.is_valid, -len(r.issues) - len(r.low_confidence_keypoints)))

    return results
