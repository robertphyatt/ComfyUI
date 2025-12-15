# Clothing Spritesheet Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a clothing spritesheet that perfectly matches a base mannequin spritesheet frame-for-frame, by finding the best matching clothed reference frame for each base frame and transforming it.

**Architecture:**
1. Validate annotations (geometric sanity + confidence checks)
2. Split input spritesheets into frames
3. For each base frame, find top 5 clothed candidates by joint distance, then pick best by pixel overlap after transform
4. Transform: scale, neck+hip align, rotate limbs, soft-edge inpaint, pixelize (factor 3)
5. Assemble outputs: clothing spritesheet, debug overlay, individual frames

**Tech Stack:** Python 3, OpenCV, NumPy, SciPy (RBFInterpolator for TPS), Matplotlib (annotator GUI)

---

## Task 1: Extend Annotation JSON Schema

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/keypoints.py`
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/annotations.py`

**Step 1: Add annotation schema helpers to new file**

Create `annotations.py` with the new per-keypoint metadata schema:

```python
"""Annotation schema and utilities for keypoint metadata tracking."""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

# New schema: each keypoint has x, y, source, confidence
# {
#   "frame.png": {
#     "image": "frame.png",
#     "keypoints": {
#       "head": {"x": 249, "y": 172, "source": "manual", "confidence": 1.0},
#       "neck": {"x": 252, "y": 217, "source": "auto", "confidence": 0.92}
#     }
#   }
# }


def migrate_legacy_annotation(keypoints: Dict) -> Dict:
    """Convert legacy [x, y] format to new {x, y, source, confidence} format.

    Legacy format: {"head": [249, 172], "neck": [252, 217]}
    New format: {"head": {"x": 249, "y": 172, "source": "auto", "confidence": 0.0}}
    """
    migrated = {}
    for name, value in keypoints.items():
        if isinstance(value, list):
            # Legacy format - assume auto with unknown confidence
            migrated[name] = {
                "x": value[0],
                "y": value[1],
                "source": "auto",
                "confidence": 0.0  # Unknown confidence from legacy
            }
        elif isinstance(value, dict) and "x" in value:
            # Already new format
            migrated[name] = value
        else:
            raise ValueError(f"Unknown keypoint format for {name}: {value}")
    return migrated


def get_keypoint_coords(keypoint: Dict) -> Tuple[int, int]:
    """Extract (x, y) coordinates from keypoint dict."""
    return (keypoint["x"], keypoint["y"])


def is_manual(keypoint: Dict) -> bool:
    """Check if keypoint was manually annotated."""
    return keypoint.get("source") == "manual"


def create_keypoint(x: int, y: int, source: str = "manual", confidence: float = 1.0) -> Dict:
    """Create a keypoint entry with full metadata."""
    return {"x": x, "y": y, "source": source, "confidence": confidence}


def load_annotations(path: Path) -> Dict[str, Dict]:
    """Load annotations, migrating legacy format if needed."""
    with open(path) as f:
        data = json.load(f)

    # Migrate each frame's keypoints if needed
    for frame_name, frame_data in data.items():
        if "keypoints" in frame_data:
            # Check if legacy format (first keypoint is a list)
            sample = next(iter(frame_data["keypoints"].values()), None)
            if isinstance(sample, list):
                frame_data["keypoints"] = migrate_legacy_annotation(frame_data["keypoints"])

    return data


def save_annotations(data: Dict[str, Dict], path: Path) -> None:
    """Save annotations to JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_coords_array(keypoints: Dict, keypoint_names: List[str]) -> List[Tuple[int, int]]:
    """Extract coordinate list from keypoints dict in order of keypoint_names."""
    coords = []
    for name in keypoint_names:
        if name in keypoints:
            kp = keypoints[name]
            if isinstance(kp, list):
                coords.append((kp[0], kp[1]))
            else:
                coords.append((kp["x"], kp["y"]))
        else:
            coords.append((0, 0))  # Missing keypoint
    return coords
```

**Step 2: Verify file created**

Run: `python3 -c "from annotations import load_annotations; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/annotations.py
git commit -m "feat: add annotation schema with source/confidence metadata"
```

---

## Task 2: Annotation Validation (Geometric + Confidence Checks)

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/validation.py`

**Step 1: Create validation module**

```python
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
```

**Step 2: Verify module loads**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from validation import validate_all_annotations; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/validation.py
git commit -m "feat: add annotation validation with geometric and confidence checks"
```

---

## Task 3: Frame Matching (Joint Distance + Pixel Overlap)

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/matching.py`

**Step 1: Create matching module**

```python
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
```

**Step 2: Verify module loads**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from matching import find_top_candidates; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/matching.py
git commit -m "feat: add frame matching by joint distance and pixel overlap"
```

---

## Task 4: Transform Pipeline (Consolidate from test_full_pipeline.py)

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`
- Reference: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/test_full_pipeline.py`

**Step 1: Create transform module**

Extract and consolidate the working transform code from test_full_pipeline.py into a clean module:

```python
"""Transform pipeline: scale, align, rotate, inpaint, pixelize."""

import cv2
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation, binary_erosion

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


@dataclass
class TransformConfig:
    """Configuration for the transform pipeline."""
    scale_factor: float = 1.057
    rotation_segment_width: int = 35
    edge_width: int = 2
    pixelize_factor: int = 3
    canvas_size: int = 512


# Limb chains for rotation (joint_idx, child_idx, name)
LIMB_CHAINS = [
    [(2, 4, "L_upper_arm"), (4, 6, "L_forearm")],   # Left arm
    [(3, 5, "R_upper_arm"), (5, 7, "R_forearm")],   # Right arm
    [(10, 12, "L_thigh"), (12, 14, "L_shin")],      # Left leg
    [(11, 13, "R_thigh"), (13, 15, "R_shin")],      # Right leg
]


def get_keypoints_array(keypoints: Dict) -> np.ndarray:
    """Convert keypoints dict to numpy array."""
    result = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    for i, name in enumerate(KEYPOINT_NAMES):
        if name in keypoints:
            kp = keypoints[name]
            if isinstance(kp, list):
                result[i] = kp
            elif isinstance(kp, dict):
                result[i] = [kp["x"], kp["y"]]
    return result


# ============ Step 1: Scale and Align (neck + hip mean) ============

def scale_and_align(
    image: np.ndarray,
    image_kpts: np.ndarray,
    target_kpts: np.ndarray,
    config: TransformConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale image and align using mean of neck and hip offsets.

    Args:
        image: Source RGBA image
        image_kpts: Keypoints for source image
        target_kpts: Target keypoints to align to
        config: Transform configuration

    Returns:
        (aligned_image, aligned_keypoints)
    """
    h, w = image.shape[:2]
    scale = config.scale_factor
    canvas_size = config.canvas_size

    # Scale
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_kpts = image_kpts * scale

    # Compute mean offset from neck (idx 1) and mid-hip
    # Mid-hip is mean of left_hip (idx 10) and right_hip (idx 11)
    neck_idx = 1
    left_hip_idx = 10
    right_hip_idx = 11

    scaled_mid_hip = (scaled_kpts[left_hip_idx] + scaled_kpts[right_hip_idx]) / 2
    target_mid_hip = (target_kpts[left_hip_idx] + target_kpts[right_hip_idx]) / 2

    neck_offset = target_kpts[neck_idx] - scaled_kpts[neck_idx]
    hip_offset = target_mid_hip - scaled_mid_hip

    # Mean offset
    mean_offset = (neck_offset + hip_offset) / 2
    offset_x = int(round(mean_offset[0]))
    offset_y = int(round(mean_offset[1]))

    # Create canvas and place scaled image
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    src_x1 = max(0, -offset_x)
    src_x2 = min(new_w, canvas_size - offset_x)
    src_y1 = max(0, -offset_y)
    src_y2 = min(new_h, canvas_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = max(0, offset_y)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

    aligned_kpts = scaled_kpts + mean_offset

    return canvas, aligned_kpts


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to extract armor from clothed image."""
    result = image.copy()
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    result[:, :, 3] = np.minimum(result[:, :, 3], mask)
    return result


# ============ Step 2: Rigid Rotation ============

def _get_bone_angle(joint: np.ndarray, child: np.ndarray) -> float:
    """Get angle of bone from joint to child."""
    delta = child - joint
    return math.atan2(delta[1], delta[0])


def _create_segment_mask(
    h: int, w: int,
    joint_pos: np.ndarray,
    child_pos: np.ndarray,
    width: int
) -> np.ndarray:
    """Create mask for a limb segment."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(child_pos[0]), int(child_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)
    return mask > 0


def _rotate_point(point: np.ndarray, pivot: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate point around pivot."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    p = point - pivot
    rotated = np.array([p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a])
    return rotated + pivot


def apply_rotation(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    target_kpts: np.ndarray,
    config: TransformConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rigid rotation at joints to match target skeleton.

    Args:
        armor: Armor RGBA image
        armor_kpts: Current armor keypoints
        target_kpts: Target keypoints to match
        config: Transform configuration

    Returns:
        (rotated_armor, rotated_keypoints)
    """
    h, w = armor.shape[:2]
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    for chain in LIMB_CHAINS:
        for joint_idx, child_idx, name in chain:
            current_angle = _get_bone_angle(result_kpts[joint_idx], result_kpts[child_idx])
            target_angle = _get_bone_angle(target_kpts[joint_idx], target_kpts[child_idx])
            delta_angle = target_angle - current_angle

            if abs(delta_angle) < 0.01:
                continue

            segment_mask = _create_segment_mask(
                h, w, result_kpts[joint_idx], result_kpts[child_idx],
                config.rotation_segment_width
            )
            armor_in_region = (result[:, :, 3] > 128) & segment_mask

            if not np.any(armor_in_region):
                continue

            pivot = result_kpts[joint_idx]

            # Clear region from result
            for c in range(4):
                result[:, :, c] = np.where(armor_in_region, 0, result[:, :, c])

            # Extract segment
            segment_img = armor.copy()
            for c in range(4):
                segment_img[:, :, c] = np.where(armor_in_region, armor[:, :, c], 0)

            # Rotate segment
            angle_deg = math.degrees(delta_angle)
            M = cv2.getRotationMatrix2D((pivot[0], pivot[1]), -angle_deg, 1.0)
            rotated_segment = cv2.warpAffine(
                segment_img, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            # Composite rotated segment back
            rot_alpha = rotated_segment[:, :, 3:4] / 255.0
            for c in range(3):
                result[:, :, c] = (rotated_segment[:, :, c] * rot_alpha[:, :, 0] +
                                  result[:, :, c] * (1 - rot_alpha[:, :, 0])).astype(np.uint8)
            result[:, :, 3] = np.maximum(result[:, :, 3], rotated_segment[:, :, 3])

            # Update keypoint
            result_kpts[child_idx] = _rotate_point(result_kpts[child_idx], pivot, delta_angle)
            armor = result.copy()

    return result, result_kpts


# ============ Step 3: Soft-Edge Inpaint ============

def _get_uncovered_mask(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int
) -> np.ndarray:
    """Get mask of base pixels not covered by armor."""
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers

    # Only below neck
    h = base_image.shape[0]
    valid_region = np.zeros((h, base_image.shape[1]), dtype=bool)
    valid_region[neck_y:, :] = True

    return uncovered & valid_region


def _get_armor_edge_near_uncovered(
    armor_alpha: np.ndarray,
    uncovered_mask: np.ndarray,
    edge_width: int
) -> np.ndarray:
    """Find armor edge pixels adjacent to uncovered areas."""
    armor_mask = armor_alpha > 128
    dilated_uncovered = binary_dilation(uncovered_mask, iterations=edge_width)
    armor_edge_near_uncovered = armor_mask & dilated_uncovered
    eroded_armor = binary_erosion(armor_mask, iterations=1)
    armor_edge = armor_mask & ~eroded_armor
    return armor_edge_near_uncovered | (armor_edge & dilated_uncovered)


def apply_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    base_image: np.ndarray,
    armor_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: TransformConfig
) -> np.ndarray:
    """Apply soft-edge texture inpainting.

    Args:
        armor: Current armor RGBA
        original_clothed: Original scaled/aligned clothed image (for texture sampling)
        base_image: Base frame RGBA
        armor_kpts: Armor keypoints (after rotation)
        base_kpts: Base keypoints
        armor_mask: Original armor mask
        config: Transform configuration

    Returns:
        Inpainted armor image
    """
    neck_y = int(base_kpts[1, 1])
    uncovered = _get_uncovered_mask(base_image, armor, neck_y)

    if not np.any(uncovered):
        return armor

    armor_edge = _get_armor_edge_near_uncovered(armor[:, :, 3], uncovered, config.edge_width)

    result = armor.copy()
    result[:, :, 3] = np.where(armor_edge, 0, armor[:, :, 3])

    inpaint_region = uncovered | armor_edge

    if not np.any(inpaint_region):
        return armor

    # TPS mapping for texture borrowing
    h, w = armor.shape[:2]
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float64)
    src_all = np.vstack([armor_kpts, corners])
    dst_all = np.vstack([base_kpts, corners])

    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0)

    inpaint_ys, inpaint_xs = np.where(inpaint_region)
    dst_coords = np.column_stack([inpaint_xs, inpaint_ys])
    src_xs = rbf_x(dst_coords)
    src_ys = rbf_y(dst_coords)

    for i, (dst_y, dst_x) in enumerate(zip(inpaint_ys, inpaint_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        # Try TPS-mapped position
        if 0 <= src_x < w and 0 <= src_y < h:
            if armor_mask[src_y, src_x] > 128:
                result[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result[dst_y, dst_x, 3] = 255
                continue

        # Fallback: nearest armor pixel
        orig_armor_alpha = armor[:, :, 3] > 128
        for radius in range(1, 30):
            y1, y2 = max(0, dst_y - radius), min(h, dst_y + radius + 1)
            x1, x2 = max(0, dst_x - radius), min(w, dst_x + radius + 1)
            box = orig_armor_alpha[y1:y2, x1:x2]
            if np.any(box):
                box_ys, box_xs = np.where(box)
                abs_ys, abs_xs = box_ys + y1, box_xs + x1
                distances = (abs_ys - dst_y) ** 2 + (abs_xs - dst_x) ** 2
                closest = np.argmin(distances)
                result[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                result[dst_y, dst_x, 3] = 255
                break

    return result


# ============ Step 4: Pixelize ============

def apply_pixelize(image: np.ndarray, factor: int) -> np.ndarray:
    """Pixelize by downscale then upscale with nearest neighbor."""
    if factor <= 1:
        return image

    h, w = image.shape[:2]
    small_h, small_w = h // factor, w // factor

    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return result


# ============ Full Pipeline ============

def transform_frame(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None
) -> np.ndarray:
    """Run full transform pipeline on a single frame.

    Args:
        clothed_image: Clothed reference frame RGBA
        clothed_kpts: Keypoints for clothed frame
        base_image: Base frame RGBA
        base_kpts: Keypoints for base frame
        armor_mask: Mask separating armor from clothed image
        config: Transform configuration (uses defaults if None)

    Returns:
        Transformed armor RGBA image
    """
    if config is None:
        config = TransformConfig()

    # Step 1: Scale and align
    aligned_clothed, aligned_kpts = scale_and_align(
        clothed_image, clothed_kpts, base_kpts, config
    )

    # Scale mask the same way
    mask_rgba = np.zeros((*armor_mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, 0] = armor_mask
    mask_rgba[:, :, 3] = armor_mask
    aligned_mask, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config)
    scaled_mask = aligned_mask[:, :, 0]

    # Extract armor
    armor = apply_mask(aligned_clothed, scaled_mask)

    # Step 2: Rotate
    rotated_armor, rotated_kpts = apply_rotation(armor, aligned_kpts, base_kpts, config)

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        rotated_armor, aligned_clothed, base_image,
        rotated_kpts, base_kpts, scaled_mask, config
    )

    # Step 4: Pixelize
    final_armor = apply_pixelize(inpainted_armor, config.pixelize_factor)

    return final_armor
```

**Step 2: Verify module loads**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from transform import transform_frame, TransformConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/transform.py
git commit -m "feat: consolidate transform pipeline (scale, align, rotate, inpaint, pixelize)"
```

---

## Task 5: Spritesheet Utilities (Split and Assemble)

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/spritesheet.py`

**Step 1: Create spritesheet module**

```python
"""Spritesheet utilities: split into frames, assemble from frames."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpritesheetLayout:
    """Layout information for a spritesheet."""
    frame_width: int
    frame_height: int
    columns: int
    rows: int
    total_frames: int


def detect_layout(spritesheet: np.ndarray) -> SpritesheetLayout:
    """Detect spritesheet layout by analyzing frame boundaries.

    Assumes:
    - All frames are same size
    - Frames are arranged in a grid
    - Transparent gaps between frames (or frame edges are detectable)

    Args:
        spritesheet: RGBA spritesheet image

    Returns:
        SpritesheetLayout with detected dimensions
    """
    h, w = spritesheet.shape[:2]
    alpha = spritesheet[:, :, 3]

    # Find vertical gaps (columns with mostly transparent pixels)
    col_alpha = np.mean(alpha, axis=0)

    # Find horizontal gaps (rows with mostly transparent pixels)
    row_alpha = np.mean(alpha, axis=1)

    # Detect frame boundaries by finding transitions
    # A frame boundary is where alpha goes from low to high or high to low
    threshold = 10  # Alpha threshold for "transparent"

    # Find frame width by detecting vertical boundaries
    in_frame = col_alpha > threshold
    transitions = np.diff(in_frame.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        # Fallback: assume single row, detect by aspect ratio
        # Common sprite sizes: 64x64, 128x128, 256x256
        for size in [64, 128, 256, 512]:
            if w % size == 0 and h % size == 0:
                cols = w // size
                rows = h // size
                return SpritesheetLayout(
                    frame_width=size,
                    frame_height=size,
                    columns=cols,
                    rows=rows,
                    total_frames=cols * rows
                )
        # Last resort: assume square frames based on height
        return SpritesheetLayout(
            frame_width=h,
            frame_height=h,
            columns=w // h,
            rows=1,
            total_frames=w // h
        )

    # Estimate frame width from detected boundaries
    frame_widths = ends - starts + 1
    if len(frame_widths) > 0:
        frame_width = int(np.median(frame_widths))
    else:
        frame_width = w

    # Similarly for height
    in_frame_row = row_alpha > threshold
    transitions_row = np.diff(in_frame_row.astype(int))
    starts_row = np.where(transitions_row == 1)[0] + 1
    ends_row = np.where(transitions_row == -1)[0]

    if len(ends_row) > 0 and len(starts_row) > 0:
        frame_heights = ends_row - starts_row + 1
        frame_height = int(np.median(frame_heights))
    else:
        frame_height = h

    columns = max(1, w // frame_width)
    rows = max(1, h // frame_height)

    return SpritesheetLayout(
        frame_width=frame_width,
        frame_height=frame_height,
        columns=columns,
        rows=rows,
        total_frames=columns * rows
    )


def split_spritesheet(
    spritesheet: np.ndarray,
    layout: Optional[SpritesheetLayout] = None
) -> List[np.ndarray]:
    """Split spritesheet into individual frames.

    Args:
        spritesheet: RGBA spritesheet image
        layout: Optional layout (auto-detected if None)

    Returns:
        List of frame images in row-major order
    """
    if layout is None:
        layout = detect_layout(spritesheet)

    frames = []
    for row in range(layout.rows):
        for col in range(layout.columns):
            x = col * layout.frame_width
            y = row * layout.frame_height
            frame = spritesheet[y:y+layout.frame_height, x:x+layout.frame_width].copy()
            frames.append(frame)

    return frames


def assemble_spritesheet(
    frames: List[np.ndarray],
    layout: SpritesheetLayout
) -> np.ndarray:
    """Assemble frames into a spritesheet.

    Args:
        frames: List of frame images
        layout: Layout specifying grid arrangement

    Returns:
        Assembled spritesheet image
    """
    sheet_h = layout.rows * layout.frame_height
    sheet_w = layout.columns * layout.frame_width

    # Determine channels from first frame
    if len(frames[0].shape) == 3:
        channels = frames[0].shape[2]
        spritesheet = np.zeros((sheet_h, sheet_w, channels), dtype=np.uint8)
    else:
        spritesheet = np.zeros((sheet_h, sheet_w), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        if idx >= layout.total_frames:
            break
        row = idx // layout.columns
        col = idx % layout.columns
        x = col * layout.frame_width
        y = row * layout.frame_height
        spritesheet[y:y+layout.frame_height, x:x+layout.frame_width] = frame

    return spritesheet


def save_frames(
    frames: List[np.ndarray],
    output_dir: Path,
    prefix: str = "frame"
) -> List[Path]:
    """Save individual frames to directory.

    Args:
        frames: List of frame images
        output_dir: Directory to save frames
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for idx, frame in enumerate(frames):
        path = output_dir / f"{prefix}_{idx:02d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)

    return paths


def load_frames(
    frame_dir: Path,
    pattern: str = "frame_*.png"
) -> List[np.ndarray]:
    """Load frames from directory.

    Args:
        frame_dir: Directory containing frames
        pattern: Glob pattern for frame files

    Returns:
        List of frame images sorted by filename
    """
    frame_dir = Path(frame_dir)
    paths = sorted(frame_dir.glob(pattern))

    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        frames.append(frame)

    return frames


def composite_overlay(
    base_frames: List[np.ndarray],
    overlay_frames: List[np.ndarray]
) -> List[np.ndarray]:
    """Composite overlay frames on top of base frames.

    Args:
        base_frames: List of base frame images
        overlay_frames: List of overlay frame images (same length as base)

    Returns:
        List of composited frames
    """
    composites = []

    for base, overlay in zip(base_frames, overlay_frames):
        result = base.copy()

        # Alpha composite
        overlay_alpha = overlay[:, :, 3:4] / 255.0
        result[:, :, :3] = (
            result[:, :, :3] * (1 - overlay_alpha) +
            overlay[:, :, :3] * overlay_alpha
        ).astype(np.uint8)
        result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])

        composites.append(result)

    return composites
```

**Step 2: Verify module loads**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from spritesheet import split_spritesheet, assemble_spritesheet; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/spritesheet.py
git commit -m "feat: add spritesheet split and assemble utilities"
```

---

## Task 6: Enhanced Annotator (Ghost Overlay + Source Display)

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/annotator.py`

**Step 1: Add ghost overlay for auto-predicted keypoints**

Add these methods to the KeypointAnnotator class and modify `__init__` and `_draw_skeleton`:

```python
# In __init__, add parameter for auto predictions:
def __init__(self, image_path: Path, existing_keypoints: Optional[Dict] = None,
             auto_predictions: Optional[Dict] = None):
    """Initialize annotator.

    Args:
        image_path: Path to sprite image
        existing_keypoints: Optional dict of existing keypoint annotations
        auto_predictions: Optional dict of auto-predicted keypoints (shown as ghosts)
    """
    self.image_path = Path(image_path)
    self.image = np.array(Image.open(image_path).convert('RGBA'))

    # Initialize keypoints: None means not yet annotated
    self.keypoints: List[Optional[Tuple[int, int]]] = [None] * NUM_KEYPOINTS
    self.keypoint_sources: List[str] = ["none"] * NUM_KEYPOINTS  # "none", "auto", "manual"

    # Store auto predictions for ghost display
    self.auto_predictions: List[Optional[Tuple[int, int]]] = [None] * NUM_KEYPOINTS

    # Load existing keypoints if provided
    if existing_keypoints:
        for i, name in enumerate(KEYPOINT_NAMES):
            if name in existing_keypoints:
                kp = existing_keypoints[name]
                if isinstance(kp, list):
                    self.keypoints[i] = tuple(kp)
                    self.keypoint_sources[i] = "auto"  # Legacy = auto
                elif isinstance(kp, dict):
                    self.keypoints[i] = (kp["x"], kp["y"])
                    self.keypoint_sources[i] = kp.get("source", "auto")

    # Load auto predictions for ghost overlay
    if auto_predictions:
        for i, name in enumerate(KEYPOINT_NAMES):
            if name in auto_predictions:
                kp = auto_predictions[name]
                if isinstance(kp, list):
                    self.auto_predictions[i] = tuple(kp)
                elif isinstance(kp, dict):
                    self.auto_predictions[i] = (kp["x"], kp["y"])

    # ... rest of init

# Modify _draw_skeleton to show ghosts and source colors:
def _draw_skeleton(self):
    """Draw all keypoints and skeleton connections."""
    # Clear existing artists
    for artist in self.point_artists + self.line_artists:
        artist.remove()
    self.point_artists = []
    self.line_artists = []

    # Draw skeleton lines (same as before)
    for (i, j), color in zip(SKELETON_CONNECTIONS, SKELETON_COLORS):
        if self.keypoints[i] is not None and self.keypoints[j] is not None:
            x = [self.keypoints[i][0], self.keypoints[j][0]]
            y = [self.keypoints[i][1], self.keypoints[j][1]]
            rgb = (color[0]/255, color[1]/255, color[2]/255)
            line, = self.ax.plot(x, y, '-', color=rgb, linewidth=2, alpha=0.8)
            self.line_artists.append(line)

    # Draw ghost predictions first (behind actual keypoints)
    for i, ghost_kp in enumerate(self.auto_predictions):
        if ghost_kp is not None and self.keypoints[i] is None:
            # Only show ghost if no actual keypoint set
            point = self.ax.scatter(ghost_kp[0], ghost_kp[1], c='yellow',
                                   s=80, marker='o', alpha=0.3,
                                   edgecolors='orange', linewidths=1,
                                   linestyle='--', zorder=5)
            self.point_artists.append(point)

    # Draw actual keypoints
    for i, kp in enumerate(self.keypoints):
        if kp is not None:
            # Color by source: green=manual, orange=auto, lime=current
            if i == self.current_keypoint_idx:
                color = 'lime'
                size = 120
            elif self.keypoint_sources[i] == "manual":
                color = 'green'
                size = 80
            else:  # auto
                color = 'orange'
                size = 60

            point = self.ax.scatter(kp[0], kp[1], c=color, s=size,
                                   marker='o', edgecolors='white',
                                   linewidths=1, zorder=10)
            self.point_artists.append(point)

            # Add label for current keypoint
            if i == self.current_keypoint_idx:
                source_label = f" [{self.keypoint_sources[i]}]"
                text = self.ax.text(kp[0] + 10, kp[1] - 10,
                                   KEYPOINT_NAMES[i] + source_label,
                                   fontsize=8, color='white',
                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                self.point_artists.append(text)

    self.fig.canvas.draw_idle()

# Modify _on_click to set source as manual:
def _on_click(self, event):
    """Handle mouse click to place keypoint."""
    if event.inaxes != self.ax:
        return
    if event.button != 1:  # Left click only
        return

    x, y = int(round(event.xdata)), int(round(event.ydata))
    self.keypoints[self.current_keypoint_idx] = (x, y)
    self.keypoint_sources[self.current_keypoint_idx] = "manual"  # Manual annotation

    # Auto-advance to next unset keypoint
    self._advance_to_next_unset()

    self.ax.set_title(self._get_title())
    self._draw_skeleton()

# Add method to accept ghost prediction:
def _accept_ghost(self):
    """Accept the ghost prediction for current keypoint."""
    ghost = self.auto_predictions[self.current_keypoint_idx]
    if ghost is not None:
        self.keypoints[self.current_keypoint_idx] = ghost
        self.keypoint_sources[self.current_keypoint_idx] = "auto"
        self._advance_to_next_unset()
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

# Modify _on_key to add 'a' for accept ghost:
def _on_key(self, event):
    """Handle keyboard input."""
    if event.key == 'right' or event.key == 'down':
        self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
    elif event.key == 'left' or event.key == 'up':
        self.current_keypoint_idx = (self.current_keypoint_idx - 1) % NUM_KEYPOINTS
    elif event.key == 's':
        self._on_save(None)
        return
    elif event.key == 'q':
        plt.close(self.fig)
        return
    elif event.key == 'c':
        self._on_clear(None)
        return
    elif event.key == 'a':  # Accept ghost
        self._accept_ghost()
        return

    self.ax.set_title(self._get_title())
    self._draw_skeleton()

# Modify run() to return keypoints with metadata:
def run(self) -> Optional[Dict[str, Dict]]:
    """Run the annotator and return keypoints if saved.

    Returns:
        Dict mapping keypoint names to {x, y, source, confidence}, or None if skipped
    """
    plt.show()

    if not self.saved:
        return None

    # Convert to dict format with metadata
    result = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        if self.keypoints[i] is not None:
            result[name] = {
                "x": self.keypoints[i][0],
                "y": self.keypoints[i][1],
                "source": self.keypoint_sources[i],
                "confidence": 1.0 if self.keypoint_sources[i] == "manual" else 0.5
            }

    return result
```

**Step 2: Update instructions text**

```python
# In _setup_gui, update the instructions:
self.fig.text(0.5, 0.97,
    'Click=place | A=accept ghost | Arrow=navigate | S=save | Q=quit | Green=manual, Orange=auto',
    ha='center', va='top', fontsize=9, color='gray')
```

**Step 3: Verify changes work**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from annotator import KeypointAnnotator; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add sprite_keypoint_detector/annotator.py
git commit -m "feat: add ghost overlay and source tracking to annotator"
```

---

## Task 7: Main Pipeline Script

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Create main pipeline orchestrator**

```python
"""Main clothing spritesheet pipeline.

Usage:
    python -m sprite_keypoint_detector.pipeline \
        --base base_spritesheet.png \
        --reference clothed_spritesheet.png \
        --output output_dir/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from .annotations import load_annotations, save_annotations, get_keypoints_array
from .validation import validate_all_annotations, ValidationResult
from .matching import find_top_candidates, score_candidate_after_transform, select_best_match, MatchCandidate, FrameMatch
from .transform import transform_frame, get_keypoints_array as get_kpts_array, TransformConfig
from .spritesheet import (
    detect_layout, split_spritesheet, assemble_spritesheet,
    save_frames, composite_overlay, SpritesheetLayout
)
from .keypoints import KEYPOINT_NAMES


class ClothingPipeline:
    """Main pipeline for generating clothing spritesheets."""

    def __init__(
        self,
        base_spritesheet_path: Path,
        reference_spritesheet_path: Path,
        annotations_path: Path,
        masks_dir: Path,
        output_dir: Path,
        config: Optional[TransformConfig] = None
    ):
        self.base_path = Path(base_spritesheet_path)
        self.reference_path = Path(reference_spritesheet_path)
        self.annotations_path = Path(annotations_path)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.config = config or TransformConfig()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)

        # Load data
        self.annotations = load_annotations(self.annotations_path)
        self.base_spritesheet = cv2.imread(str(self.base_path), cv2.IMREAD_UNCHANGED)
        self.reference_spritesheet = cv2.imread(str(self.reference_path), cv2.IMREAD_UNCHANGED)

        # Detect layout from base
        self.layout = detect_layout(self.base_spritesheet)
        print(f"Detected layout: {self.layout.columns}x{self.layout.rows} frames, "
              f"{self.layout.frame_width}x{self.layout.frame_height}px each")

        # Split spritesheets
        self.base_frames = split_spritesheet(self.base_spritesheet, self.layout)
        self.reference_frames = split_spritesheet(self.reference_spritesheet, self.layout)

        print(f"Split {len(self.base_frames)} base frames, {len(self.reference_frames)} reference frames")

    def validate_annotations(self) -> List[ValidationResult]:
        """Validate all annotations, return flagged frames."""
        print("\n=== Validating Annotations ===")
        results = validate_all_annotations(self.annotations)

        flagged = [r for r in results if not r.is_valid]
        print(f"Total frames: {len(results)}")
        print(f"Flagged for review: {len(flagged)}")

        for r in flagged:
            print(f"  {r.frame_name}:")
            for issue in r.issues:
                print(f"    - {issue}")
            for lc in r.low_confidence_keypoints:
                print(f"    - Low confidence: {lc}")

        return flagged

    def match_frames(self, blue_threshold: int = 2000) -> List[FrameMatch]:
        """Match each base frame to best clothed frame."""
        print("\n=== Matching Frames ===")

        # Separate base and clothed annotations
        base_annotations = {k: v for k, v in self.annotations.items() if k.startswith("base_")}
        clothed_annotations = {k: v for k, v in self.annotations.items() if k.startswith("clothed_")}

        matches = []

        for base_idx, (base_name, base_data) in enumerate(sorted(base_annotations.items())):
            print(f"\nMatching {base_name} ({base_idx + 1}/{len(base_annotations)})")

            base_kpts = base_data.get("keypoints", {})
            base_frame = self.base_frames[base_idx]

            # Find top 5 candidates by joint distance
            candidates = find_top_candidates(base_name, base_kpts, clothed_annotations, top_n=5)
            print(f"  Top 5 by joint distance: {[c[0] for c in candidates]}")

            # Score each candidate after transform
            scored_candidates = []
            for clothed_name, joint_dist in candidates:
                clothed_idx = int(clothed_name.split("_")[-1].replace(".png", ""))
                clothed_frame = self.reference_frames[clothed_idx]
                clothed_kpts = clothed_annotations[clothed_name].get("keypoints", {})

                # Load mask
                mask_path = self.masks_dir / f"mask_{clothed_idx:02d}.png"
                if not mask_path.exists():
                    print(f"    Warning: mask not found for {clothed_name}")
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                # Transform and score
                clothed_kpts_array = get_kpts_array(clothed_kpts)
                base_kpts_array = get_kpts_array(base_kpts)

                transformed = transform_frame(
                    clothed_frame, clothed_kpts_array,
                    base_frame, base_kpts_array,
                    mask, self.config
                )

                neck_y = int(base_kpts_array[1, 1])
                blue, red = score_candidate_after_transform(base_frame, transformed, neck_y)

                scored_candidates.append(MatchCandidate(
                    clothed_frame=clothed_name,
                    joint_distance=joint_dist,
                    blue_pixels=blue,
                    red_pixels=red,
                    score_rank=0
                ))
                print(f"    {clothed_name}: blue={blue}, red={red}")

            # Select best
            if scored_candidates:
                best, needs_review = select_best_match(scored_candidates, blue_threshold)

                # Update ranks
                sorted_by_score = sorted(scored_candidates, key=lambda c: (c.blue_pixels, c.red_pixels))
                for rank, c in enumerate(sorted_by_score):
                    c.score_rank = rank + 1

                match = FrameMatch(
                    base_frame=base_name,
                    matched_clothed_frame=best.clothed_frame,
                    candidates=scored_candidates,
                    needs_review=needs_review
                )
                matches.append(match)

                status = "NEEDS REVIEW" if needs_review else "OK"
                print(f"  -> Best match: {best.clothed_frame} (blue={best.blue_pixels}) [{status}]")

        return matches

    def generate_outputs(self, matches: List[FrameMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate final clothing and debug overlay spritesheets."""
        print("\n=== Generating Outputs ===")

        clothed_annotations = {k: v for k, v in self.annotations.items() if k.startswith("clothed_")}
        base_annotations = {k: v for k, v in self.annotations.items() if k.startswith("base_")}

        clothing_frames = []

        for match in matches:
            base_idx = int(match.base_frame.split("_")[-1].replace(".png", ""))
            clothed_idx = int(match.matched_clothed_frame.split("_")[-1].replace(".png", ""))

            base_frame = self.base_frames[base_idx]
            clothed_frame = self.reference_frames[clothed_idx]

            base_kpts = get_kpts_array(base_annotations[match.base_frame].get("keypoints", {}))
            clothed_kpts = get_kpts_array(clothed_annotations[match.matched_clothed_frame].get("keypoints", {}))

            # Load mask
            mask_path = self.masks_dir / f"mask_{clothed_idx:02d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            # Transform
            transformed = transform_frame(
                clothed_frame, clothed_kpts,
                base_frame, base_kpts,
                mask, self.config
            )

            clothing_frames.append(transformed)
            print(f"  Generated frame {base_idx:02d} from {match.matched_clothed_frame}")

        # Save individual frames
        save_frames(clothing_frames, self.output_dir / "frames", prefix="clothing")

        # Assemble clothing spritesheet
        clothing_sheet = assemble_spritesheet(clothing_frames, self.layout)
        cv2.imwrite(str(self.output_dir / "clothing.png"), clothing_sheet)
        print(f"Saved: {self.output_dir / 'clothing.png'}")

        # Create debug overlay
        overlay_frames = composite_overlay(self.base_frames, clothing_frames)
        overlay_sheet = assemble_spritesheet(overlay_frames, self.layout)
        cv2.imwrite(str(self.output_dir / "debug_overlay.png"), overlay_sheet)
        print(f"Saved: {self.output_dir / 'debug_overlay.png'}")

        return clothing_sheet, overlay_sheet

    def run(self, skip_validation: bool = False) -> bool:
        """Run the full pipeline.

        Args:
            skip_validation: Skip annotation validation step

        Returns:
            True if successful, False if manual intervention needed
        """
        # Step 1: Validate annotations
        if not skip_validation:
            flagged = self.validate_annotations()
            if flagged:
                print(f"\n{len(flagged)} frames need manual review before proceeding.")
                print("Run with --skip-validation to proceed anyway, or fix annotations first.")
                return False

        # Step 2: Match frames
        matches = self.match_frames()

        # Check for frames needing review
        needs_review = [m for m in matches if m.needs_review]
        if needs_review:
            print(f"\n{len(needs_review)} matches need manual review:")
            for m in needs_review:
                print(f"  {m.base_frame} -> {m.matched_clothed_frame}")
            print("Proceeding with best available matches...")

        # Step 3: Generate outputs
        self.generate_outputs(matches)

        print("\n=== Pipeline Complete ===")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate clothing spritesheet from base and reference"
    )
    parser.add_argument("--base", type=Path, required=True,
                       help="Base mannequin spritesheet")
    parser.add_argument("--reference", type=Path, required=True,
                       help="Clothed reference spritesheet")
    parser.add_argument("--annotations", type=Path, required=True,
                       help="Annotations JSON file")
    parser.add_argument("--masks", type=Path, required=True,
                       help="Directory containing armor masks")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip annotation validation")
    parser.add_argument("--scale", type=float, default=1.057,
                       help="Scale factor for clothed frames")
    parser.add_argument("--pixelize", type=int, default=3,
                       help="Pixelization factor (1=none)")

    args = parser.parse_args()

    config = TransformConfig(
        scale_factor=args.scale,
        pixelize_factor=args.pixelize
    )

    pipeline = ClothingPipeline(
        base_spritesheet_path=args.base,
        reference_spritesheet_path=args.reference,
        annotations_path=args.annotations,
        masks_dir=args.masks,
        output_dir=args.output,
        config=config
    )

    success = pipeline.run(skip_validation=args.skip_validation)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
```

**Step 2: Verify module loads**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from pipeline import ClothingPipeline; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/pipeline.py
git commit -m "feat: add main pipeline orchestrator"
```

---

## Task 8: Annotation Utilities (Retrain, Re-annotate, Manual Edit)

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/annotation_utils.py`

**Step 1: Create annotation utilities**

```python
"""Annotation utilities: retrain model, re-annotate, manual edit.

Three utilities:
1. retrain - Train model on manually confirmed annotations
2. reannotate - Run model on all frames, update only auto keypoints
3. edit - Manual annotation editor with ghost predictions
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import json

from .annotations import load_annotations, save_annotations, is_manual
from .annotator import KeypointAnnotator, annotate_directory
from .keypoints import KEYPOINT_NAMES
from .inference import predict_keypoints  # Assumes this exists
from .train import train_model  # Assumes this exists


def retrain_on_manual(
    annotations_path: Path,
    frames_dir: Path,
    model_output_path: Path
) -> None:
    """Retrain keypoint model on manually confirmed annotations only.

    Args:
        annotations_path: Path to annotations JSON
        frames_dir: Directory containing frame images
        model_output_path: Path to save trained model
    """
    annotations = load_annotations(annotations_path)

    # Filter to only manually annotated keypoints
    manual_annotations = {}
    for frame_name, frame_data in annotations.items():
        keypoints = frame_data.get("keypoints", {})
        manual_kpts = {}
        for name, kp in keypoints.items():
            if isinstance(kp, dict) and kp.get("source") == "manual":
                manual_kpts[name] = [kp["x"], kp["y"]]
            elif isinstance(kp, list):
                # Legacy format - skip (unknown source)
                pass

        if manual_kpts:
            manual_annotations[frame_name] = {
                "image": frame_name,
                "keypoints": manual_kpts
            }

    print(f"Found {len(manual_annotations)} frames with manual annotations")

    if len(manual_annotations) < 5:
        print("Warning: Very few manual annotations. Model may not train well.")

    # Save filtered annotations for training
    train_annotations_path = annotations_path.parent / "train_annotations.json"
    with open(train_annotations_path, 'w') as f:
        json.dump(manual_annotations, f, indent=2)

    # Train model
    train_model(
        annotations_path=train_annotations_path,
        frames_dir=frames_dir,
        output_path=model_output_path
    )

    print(f"Model saved to {model_output_path}")


def reannotate_auto(
    annotations_path: Path,
    frames_dir: Path,
    model_path: Path
) -> None:
    """Re-run automatic annotation, preserving manual keypoints.

    Args:
        annotations_path: Path to annotations JSON
        frames_dir: Directory containing frame images
        model_path: Path to trained model
    """
    annotations = load_annotations(annotations_path)

    # Get all frame images
    frame_files = sorted(frames_dir.glob("*.png"))

    updated_count = 0
    for frame_path in frame_files:
        frame_name = frame_path.name

        # Run inference
        predictions = predict_keypoints(frame_path, model_path)

        # Get existing keypoints
        existing = annotations.get(frame_name, {}).get("keypoints", {})

        # Update only non-manual keypoints
        updated_kpts = {}
        for name in KEYPOINT_NAMES:
            if name in existing:
                kp = existing[name]
                if isinstance(kp, dict) and kp.get("source") == "manual":
                    # Keep manual annotation
                    updated_kpts[name] = kp
                    continue

            # Use prediction
            if name in predictions:
                pred = predictions[name]
                updated_kpts[name] = {
                    "x": pred["x"],
                    "y": pred["y"],
                    "source": "auto",
                    "confidence": pred.get("confidence", 0.5)
                }

        if frame_name not in annotations:
            annotations[frame_name] = {"image": frame_name}

        annotations[frame_name]["keypoints"] = updated_kpts
        updated_count += 1

    save_annotations(annotations, annotations_path)
    print(f"Updated {updated_count} frames with auto predictions (manual preserved)")


def edit_frame(
    frame_path: Path,
    annotations_path: Path,
    model_path: Optional[Path] = None
) -> None:
    """Manually edit annotations for a single frame with ghost predictions.

    Args:
        frame_path: Path to frame image
        annotations_path: Path to annotations JSON
        model_path: Optional model path for ghost predictions
    """
    annotations = load_annotations(annotations_path)
    frame_name = frame_path.name

    existing = annotations.get(frame_name, {}).get("keypoints", {})

    # Get auto predictions for ghost overlay
    auto_predictions = None
    if model_path and model_path.exists():
        auto_predictions = predict_keypoints(frame_path, model_path)

    # Run annotator
    annotator = KeypointAnnotator(frame_path, existing, auto_predictions)
    result = annotator.run()

    if result:
        if frame_name not in annotations:
            annotations[frame_name] = {"image": frame_name}
        annotations[frame_name]["keypoints"] = result
        save_annotations(annotations, annotations_path)
        print(f"Saved annotations for {frame_name}")
    else:
        print("Skipped (no changes saved)")


def edit_flagged(
    annotations_path: Path,
    frames_dir: Path,
    model_path: Optional[Path] = None
) -> None:
    """Edit all flagged frames (low confidence or validation issues).

    Args:
        annotations_path: Path to annotations JSON
        frames_dir: Directory containing frame images
        model_path: Optional model path for ghost predictions
    """
    from .validation import validate_all_annotations

    annotations = load_annotations(annotations_path)
    results = validate_all_annotations(annotations)

    flagged = [r for r in results if not r.is_valid]

    if not flagged:
        print("No flagged frames to edit!")
        return

    print(f"Found {len(flagged)} flagged frames to review")

    for i, result in enumerate(flagged):
        frame_name = result.frame_name
        frame_path = frames_dir / frame_name

        if not frame_path.exists():
            print(f"Warning: {frame_path} not found, skipping")
            continue

        print(f"\n[{i+1}/{len(flagged)}] {frame_name}")
        print(f"  Issues: {result.issues}")
        print(f"  Low confidence: {result.low_confidence_keypoints}")

        resp = input("  Edit this frame? (Y/n/q): ").strip().lower()
        if resp == 'q':
            print("Quitting review")
            break
        if resp == 'n':
            continue

        edit_frame(frame_path, annotations_path, model_path)


def main():
    parser = argparse.ArgumentParser(description="Annotation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Retrain command
    retrain_parser = subparsers.add_parser("retrain", help="Retrain model on manual annotations")
    retrain_parser.add_argument("--annotations", type=Path, required=True)
    retrain_parser.add_argument("--frames", type=Path, required=True)
    retrain_parser.add_argument("--output", type=Path, required=True)

    # Reannotate command
    reann_parser = subparsers.add_parser("reannotate", help="Re-run auto annotation")
    reann_parser.add_argument("--annotations", type=Path, required=True)
    reann_parser.add_argument("--frames", type=Path, required=True)
    reann_parser.add_argument("--model", type=Path, required=True)

    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Manual annotation editor")
    edit_parser.add_argument("--frame", type=Path, help="Single frame to edit")
    edit_parser.add_argument("--flagged", action="store_true", help="Edit all flagged frames")
    edit_parser.add_argument("--annotations", type=Path, required=True)
    edit_parser.add_argument("--frames", type=Path, required=True)
    edit_parser.add_argument("--model", type=Path, help="Model for ghost predictions")

    args = parser.parse_args()

    if args.command == "retrain":
        retrain_on_manual(args.annotations, args.frames, args.output)
    elif args.command == "reannotate":
        reannotate_auto(args.annotations, args.frames, args.model)
    elif args.command == "edit":
        if args.flagged:
            edit_flagged(args.annotations, args.frames, args.model)
        elif args.frame:
            edit_frame(args.frame, args.annotations, args.model)
        else:
            print("Specify --frame or --flagged")


if __name__ == "__main__":
    main()
```

**Step 2: Verify module loads**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from annotation_utils import edit_frame; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/annotation_utils.py
git commit -m "feat: add annotation utilities (retrain, reannotate, edit)"
```

---

## Task 9: Integration Test

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/tests/test_pipeline_integration.py`

**Step 1: Create integration test**

```python
"""Integration test for the clothing pipeline."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json

from ..transform import transform_frame, TransformConfig, get_keypoints_array
from ..spritesheet import split_spritesheet, assemble_spritesheet, detect_layout
from ..matching import compute_joint_distance, find_top_candidates
from ..validation import validate_frame, compute_median_bone_lengths


class TestTransformPipeline:
    """Test the transform pipeline components."""

    def test_transform_produces_output(self):
        """Transform should produce non-empty output."""
        # Create simple test images
        clothed = np.zeros((512, 512, 4), dtype=np.uint8)
        clothed[200:300, 200:300, :3] = 128  # Gray square
        clothed[200:300, 200:300, 3] = 255   # Opaque

        base = np.zeros((512, 512, 4), dtype=np.uint8)
        base[210:310, 210:310, :3] = 100
        base[210:310, 210:310, 3] = 255

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[200:300, 200:300] = 255

        # Simple keypoints (just need neck for alignment)
        clothed_kpts = np.zeros((18, 2))
        clothed_kpts[1] = [250, 200]  # neck
        clothed_kpts[10] = [240, 300]  # left hip
        clothed_kpts[11] = [260, 300]  # right hip

        base_kpts = np.zeros((18, 2))
        base_kpts[1] = [260, 210]  # neck (offset)
        base_kpts[10] = [250, 310]
        base_kpts[11] = [270, 310]

        config = TransformConfig(scale_factor=1.0, pixelize_factor=1)

        result = transform_frame(clothed, clothed_kpts, base, base_kpts, mask, config)

        assert result is not None
        assert result.shape == (512, 512, 4)
        assert np.any(result[:, :, 3] > 0)  # Has some visible pixels


class TestSpritesheet:
    """Test spritesheet utilities."""

    def test_split_and_assemble_roundtrip(self):
        """Split then assemble should produce same image."""
        # Create test spritesheet (2x2 grid of 64x64 frames)
        sheet = np.zeros((128, 128, 4), dtype=np.uint8)
        sheet[0:64, 0:64, 0] = 255    # Red frame
        sheet[0:64, 64:128, 1] = 255  # Green frame
        sheet[64:128, 0:64, 2] = 255  # Blue frame
        sheet[64:128, 64:128, :3] = 128  # Gray frame
        sheet[:, :, 3] = 255  # All opaque

        from ..spritesheet import SpritesheetLayout
        layout = SpritesheetLayout(
            frame_width=64, frame_height=64,
            columns=2, rows=2, total_frames=4
        )

        frames = split_spritesheet(sheet, layout)
        assert len(frames) == 4

        reassembled = assemble_spritesheet(frames, layout)
        assert np.array_equal(sheet, reassembled)


class TestMatching:
    """Test frame matching."""

    def test_joint_distance_identical(self):
        """Identical keypoints should have zero distance."""
        kpts = {"head": [100, 100], "neck": [100, 150]}
        dist = compute_joint_distance(kpts, kpts, ["head", "neck"])
        assert dist == 0.0

    def test_joint_distance_different(self):
        """Different keypoints should have positive distance."""
        kpts1 = {"head": [100, 100], "neck": [100, 150]}
        kpts2 = {"head": [110, 100], "neck": [100, 160]}
        dist = compute_joint_distance(kpts1, kpts2, ["head", "neck"])
        assert dist > 0


class TestValidation:
    """Test annotation validation."""

    def test_validate_good_frame(self):
        """Valid keypoints should pass validation."""
        kpts = {
            "head": {"x": 256, "y": 100, "source": "manual", "confidence": 1.0},
            "neck": {"x": 256, "y": 150, "source": "manual", "confidence": 1.0},
            "left_shoulder": {"x": 200, "y": 170, "source": "manual", "confidence": 1.0},
            "right_shoulder": {"x": 312, "y": 170, "source": "manual", "confidence": 1.0},
        }
        result = validate_frame("test.png", kpts)
        assert result.is_valid

    def test_validate_out_of_bounds(self):
        """Out of bounds keypoints should fail."""
        kpts = {
            "head": {"x": 600, "y": 100, "source": "auto", "confidence": 0.9},
        }
        result = validate_frame("test.png", kpts, image_bounds=(512, 512))
        assert not result.is_valid
        assert any("outside bounds" in issue for issue in result.issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Run tests**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -m pytest tests/test_pipeline_integration.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/tests/test_pipeline_integration.py
git commit -m "test: add pipeline integration tests"
```

---

## Summary

The implementation plan creates a modular clothing spritesheet pipeline with:

1. **annotations.py** - Schema with source/confidence metadata
2. **validation.py** - Geometric + confidence sanity checks
3. **matching.py** - Joint distance + pixel overlap scoring
4. **transform.py** - Scale, align, rotate, inpaint, pixelize
5. **spritesheet.py** - Split and assemble utilities
6. **annotator.py** - Enhanced with ghost overlay
7. **pipeline.py** - Main orchestrator
8. **annotation_utils.py** - Retrain, reannotate, edit utilities
9. **tests/** - Integration tests

**Key parameters:**
- Scale factor: 1.057
- Pixelize factor: 3
- Blue threshold: 2000 (triggers manual review)
- Top candidates: 5
- Alignment: mean of neck + hip offset

---

Plan complete and saved to `docs/plans/2025-12-15-clothing-spritesheet-pipeline.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?