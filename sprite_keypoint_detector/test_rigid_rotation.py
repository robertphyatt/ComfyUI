#!/usr/bin/env python3
"""Rigid rotation transform at joints to match base skeleton.

Pipeline:
1. Scale and align armor to base (neck-based)
2. For each limb segment, rotate around joint to match target bone angle
3. Measure blue (uncovered base) + red (floating armor) to verify improvement
4. Soft-edge inpaint remaining gaps
5. Pixelize

Key insight: Rotation preserves armor shape perfectly - no distortion.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import json
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.interpolate import RBFInterpolator
import math

sys.path.insert(0, str(Path(__file__).parent))
from keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


FRAME_MAPPINGS = {
    "clothed_frame_00": "base_frame_23",
    "clothed_frame_01": "base_frame_24",
}


@dataclass
class OptimizerConfig:
    scale_factor: float = 1.057


# Limb chains: each chain is processed root-to-tip
# Format: (joint_idx, child_idx, name)
# The joint is the pivot point for rotation
LIMB_CHAINS = [
    # Left arm: shoulder -> elbow -> wrist
    [(2, 4, "L_upper_arm"), (4, 6, "L_forearm")],
    # Right arm: shoulder -> elbow -> wrist
    [(3, 5, "R_upper_arm"), (5, 7, "R_forearm")],
    # Left leg: hip -> knee -> ankle
    [(10, 12, "L_thigh"), (12, 14, "L_shin")],
    # Right leg: hip -> knee -> ankle
    [(11, 13, "R_thigh"), (13, 15, "R_shin")],
]


def load_annotations(annotations_path: Path) -> Dict[str, Dict]:
    with open(annotations_path) as f:
        return json.load(f)


def get_keypoints_array(annotations: Dict, frame_name: str) -> np.ndarray:
    key = f"{frame_name}.png"
    kpts = annotations[key]["keypoints"]
    result = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    for i, name in enumerate(KEYPOINT_NAMES):
        if name in kpts:
            result[i] = kpts[name]
    return result


def scale_and_align_image(
    image: np.ndarray,
    image_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    scale_factor: float,
    canvas_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_kpts = image_keypoints * scale_factor

    neck_offset = target_keypoints[1] - scaled_kpts[1]
    offset_x = int(round(neck_offset[0]))
    offset_y = int(round(neck_offset[1]))

    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    src_x1 = max(0, -offset_x)
    src_x2 = min(new_w, canvas_size - offset_x)
    src_y1 = max(0, -offset_y)
    src_y2 = min(new_h, canvas_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = max(0, offset_y)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]
    aligned_kpts = scaled_kpts + neck_offset

    return canvas, aligned_kpts


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    result[:, :, 3] = np.minimum(result[:, :, 3], mask)
    return result


def get_bone_angle(joint: np.ndarray, child: np.ndarray) -> float:
    """Get angle of bone from joint to child in radians."""
    delta = child - joint
    return math.atan2(delta[1], delta[0])


def create_segment_mask(
    h: int, w: int,
    joint_pos: np.ndarray,
    child_pos: np.ndarray,
    width: int = 35
) -> np.ndarray:
    """Create mask for pixels belonging to this segment."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(child_pos[0]), int(child_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)
    return mask > 0


def rotate_segment(
    image: np.ndarray,
    segment_mask: np.ndarray,
    pivot: np.ndarray,
    angle_rad: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate segment pixels around pivot point.

    Returns: (rotated_image, rotated_segment_mask)
    """
    h, w = image.shape[:2]

    # Create rotation matrix
    angle_deg = math.degrees(angle_rad)
    M = cv2.getRotationMatrix2D((pivot[0], pivot[1]), -angle_deg, 1.0)

    # Extract segment pixels
    segment_img = np.zeros_like(image)
    for c in range(4):
        segment_img[:, :, c] = np.where(segment_mask, image[:, :, c], 0)

    # Rotate segment
    rotated_segment = cv2.warpAffine(
        segment_img, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Rotate mask
    mask_uint8 = segment_mask.astype(np.uint8) * 255
    rotated_mask = cv2.warpAffine(
        mask_uint8, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ) > 128

    return rotated_segment, rotated_mask


def rotate_point(point: np.ndarray, pivot: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point around a pivot."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Translate to origin
    p = point - pivot

    # Rotate
    rotated = np.array([
        p[0] * cos_a - p[1] * sin_a,
        p[0] * sin_a + p[1] * cos_a
    ])

    # Translate back
    return rotated + pivot


def apply_chain_rotations(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    target_kpts: np.ndarray,
    chain: List[Tuple[int, int, str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rotations to a limb chain to match target skeleton.

    Returns: (transformed_armor, updated_armor_keypoints)
    """
    h, w = armor.shape[:2]
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    # Process chain from root to tip
    for joint_idx, child_idx, name in chain:
        # Current bone angle (in armor)
        current_angle = get_bone_angle(result_kpts[joint_idx], result_kpts[child_idx])

        # Target bone angle
        target_angle = get_bone_angle(target_kpts[joint_idx], target_kpts[child_idx])

        # Angle difference
        delta_angle = target_angle - current_angle

        # Skip if angle is very small
        if abs(delta_angle) < 0.01:  # ~0.5 degrees
            continue

        # Create mask for this segment and all downstream segments
        # For now, just use this segment
        segment_mask = create_segment_mask(
            h, w, result_kpts[joint_idx], result_kpts[child_idx], width=35
        )

        # Also include armor pixels in this region
        armor_in_region = (result[:, :, 3] > 128) & segment_mask

        if not np.any(armor_in_region):
            continue

        # Pivot point is the joint
        pivot = result_kpts[joint_idx]

        # Remove this segment from result
        for c in range(4):
            result[:, :, c] = np.where(armor_in_region, 0, result[:, :, c])

        # Rotate segment
        segment_img = armor.copy()
        for c in range(4):
            segment_img[:, :, c] = np.where(armor_in_region, armor[:, :, c], 0)

        # Apply rotation
        angle_deg = math.degrees(delta_angle)
        M = cv2.getRotationMatrix2D((pivot[0], pivot[1]), -angle_deg, 1.0)

        rotated_segment = cv2.warpAffine(
            segment_img, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Composite rotated segment onto result
        rot_alpha = rotated_segment[:, :, 3:4] / 255.0
        for c in range(3):
            result[:, :, c] = (rotated_segment[:, :, c] * rot_alpha[:, :, 0] +
                              result[:, :, c] * (1 - rot_alpha[:, :, 0])).astype(np.uint8)
        result[:, :, 3] = np.maximum(result[:, :, 3], rotated_segment[:, :, 3])

        # Update keypoints - rotate child (and downstream) around joint
        result_kpts[child_idx] = rotate_point(result_kpts[child_idx], pivot, delta_angle)

        # Also update the armor source for next iteration
        armor = result.copy()

        print(f"    {name}: rotated {math.degrees(delta_angle):.1f}°")

    return result, result_kpts


def count_blue_red(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int
) -> Tuple[int, int]:
    """Count blue (uncovered base) and red (floating armor) pixels."""
    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128

    h = base_image.shape[0]
    valid = np.zeros((h, base_image.shape[1]), dtype=bool)
    valid[neck_y:, :] = True

    blue = base_visible & ~armor_visible & valid  # base not covered
    red = armor_visible & ~base_visible & valid   # armor floating

    return int(np.sum(blue)), int(np.sum(red))


def composite_on_base(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    result = base.copy()
    mask = overlay[:, :, 3:4] / 255.0
    result[:, :, :3] = (result[:, :, :3] * (1 - mask) + overlay[:, :, :3] * mask).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])
    return result


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    result = img.copy()
    cv2.putText(result, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255, 255), 1)
    return result


def main():
    base_dir = Path(__file__).parent.parent / "training_data"
    frames_dir = base_dir / "frames"
    masks_dir = base_dir / "masks_corrected"
    annotations_path = base_dir / "annotations.json"
    output_dir = base_dir / "skeleton_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_path)
    config = OptimizerConfig()

    for clothed_name, base_name in FRAME_MAPPINGS.items():
        print(f"\n=== Processing {clothed_name} -> {base_name} ===")

        clothed_img = cv2.imread(str(frames_dir / f"{clothed_name}.png"), cv2.IMREAD_UNCHANGED)
        base_img = cv2.imread(str(frames_dir / f"{base_name}.png"), cv2.IMREAD_UNCHANGED)

        mask_idx = clothed_name.split("_")[-1]
        mask_img = cv2.imread(str(masks_dir / f"mask_{mask_idx}.png"), cv2.IMREAD_UNCHANGED)
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]

        clothed_kpts = get_keypoints_array(annotations, clothed_name)
        base_kpts = get_keypoints_array(annotations, base_name)

        # Step 1: Scale and align
        print("\nStep 1: Scale and align...")
        scaled_clothed, scaled_kpts = scale_and_align_image(
            clothed_img, clothed_kpts, base_kpts, config.scale_factor
        )
        mask_rgba = np.zeros((*mask_img.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = mask_img
        mask_rgba[:, :, 3] = mask_img
        scaled_mask, _ = scale_and_align_image(
            mask_rgba, clothed_kpts, base_kpts, config.scale_factor
        )
        armor = apply_mask_to_image(scaled_clothed, scaled_mask[:, :, 0])

        neck_y = int(base_kpts[1, 1])

        blue_before, red_before = count_blue_red(base_img, armor, neck_y)
        print(f"After scale/align: blue={blue_before}, red={red_before}")

        # Step 2: Rigid rotations at joints
        print("\nStep 2: Rigid rotations at joints...")
        rotated_armor = armor.copy()
        rotated_kpts = scaled_kpts.copy()

        for chain in LIMB_CHAINS:
            rotated_armor, rotated_kpts = apply_chain_rotations(
                rotated_armor, rotated_kpts, base_kpts, chain
            )

        blue_after, red_after = count_blue_red(base_img, rotated_armor, neck_y)
        print(f"After rotations: blue={blue_after}, red={red_after}")
        print(f"Improvement: blue {blue_before - blue_after:+d}, red {red_before - red_after:+d}")

        # Create comparison
        before_comp = composite_on_base(base_img, armor)
        after_comp = composite_on_base(base_img, rotated_armor)

        before_l = add_label(before_comp, f"before: b={blue_before} r={red_before}")
        after_l = add_label(after_comp, f"after: b={blue_after} r={red_after}")

        comparison = np.hstack([before_l, after_l])

        cv2.imwrite(str(output_dir / f"{clothed_name}_rotation.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_rotated_armor.png"), after_comp)
        print(f"Saved: {clothed_name}_rotation.png")


if __name__ == "__main__":
    main()
