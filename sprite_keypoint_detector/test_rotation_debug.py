#!/usr/bin/env python3
"""Debug visualization for rotation transforms.

Shows:
1. Blue/green/red overlay (uncovered/covering/floating)
2. Skeleton keypoints for both armor and base
3. Before and after rotation
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json
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


# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1),   # head -> neck
    (1, 2),   # neck -> L_shoulder
    (1, 3),   # neck -> R_shoulder
    (2, 4),   # L_shoulder -> L_elbow
    (4, 6),   # L_elbow -> L_wrist
    (3, 5),   # R_shoulder -> R_elbow
    (5, 7),   # R_elbow -> R_wrist
    (1, 8),   # neck -> mid_hip
    (8, 10),  # mid_hip -> L_hip
    (8, 11),  # mid_hip -> R_hip
    (10, 12), # L_hip -> L_knee
    (12, 14), # L_knee -> L_ankle
    (11, 13), # R_hip -> R_knee
    (13, 15), # R_knee -> R_ankle
]

LIMB_CHAINS = [
    [(2, 4, "L_upper_arm"), (4, 6, "L_forearm")],
    [(3, 5, "R_upper_arm"), (5, 7, "R_forearm")],
    [(10, 12, "L_thigh"), (12, 14, "L_shin")],
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
    delta = child - joint
    return math.atan2(delta[1], delta[0])


def create_segment_mask(h: int, w: int, joint_pos: np.ndarray, child_pos: np.ndarray, width: int = 35) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(child_pos[0]), int(child_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)
    return mask > 0


def rotate_point(point: np.ndarray, pivot: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    p = point - pivot
    rotated = np.array([p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a])
    return rotated + pivot


def apply_chain_rotations(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    target_kpts: np.ndarray,
    chain: List[Tuple[int, int, str]]
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = armor.shape[:2]
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    for joint_idx, child_idx, name in chain:
        current_angle = get_bone_angle(result_kpts[joint_idx], result_kpts[child_idx])
        target_angle = get_bone_angle(target_kpts[joint_idx], target_kpts[child_idx])
        delta_angle = target_angle - current_angle

        if abs(delta_angle) < 0.01:
            continue

        segment_mask = create_segment_mask(h, w, result_kpts[joint_idx], result_kpts[child_idx], width=35)
        armor_in_region = (result[:, :, 3] > 128) & segment_mask

        if not np.any(armor_in_region):
            continue

        pivot = result_kpts[joint_idx]

        for c in range(4):
            result[:, :, c] = np.where(armor_in_region, 0, result[:, :, c])

        segment_img = armor.copy()
        for c in range(4):
            segment_img[:, :, c] = np.where(armor_in_region, armor[:, :, c], 0)

        angle_deg = math.degrees(delta_angle)
        M = cv2.getRotationMatrix2D((pivot[0], pivot[1]), -angle_deg, 1.0)

        rotated_segment = cv2.warpAffine(
            segment_img, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        rot_alpha = rotated_segment[:, :, 3:4] / 255.0
        for c in range(3):
            result[:, :, c] = (rotated_segment[:, :, c] * rot_alpha[:, :, 0] +
                              result[:, :, c] * (1 - rot_alpha[:, :, 0])).astype(np.uint8)
        result[:, :, 3] = np.maximum(result[:, :, 3], rotated_segment[:, :, 3])

        result_kpts[child_idx] = rotate_point(result_kpts[child_idx], pivot, delta_angle)
        armor = result.copy()

    return result, result_kpts


def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int], thickness: int = 2):
    """Draw skeleton on image."""
    result = img.copy()

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(result, pt1, pt2, color + (255,), thickness)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(result, (int(x), int(y)), 3, color + (255,), -1)

    return result


def create_overlap_viz(base_image: np.ndarray, armor_image: np.ndarray, neck_y: int) -> np.ndarray:
    """Create blue/green/red visualization."""
    h, w = base_image.shape[:2]

    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128

    valid = np.zeros((h, w), dtype=bool)
    valid[neck_y:, :] = True

    # Categories
    green = armor_visible & base_visible & valid   # armor covering base (good)
    red = armor_visible & ~base_visible & valid    # armor floating
    blue = base_visible & ~armor_visible & valid   # base uncovered

    # Start with composite
    vis = base_image.copy()
    armor_alpha = armor_image[:, :, 3:4] / 255.0
    vis[:, :, :3] = (vis[:, :, :3] * (1 - armor_alpha) + armor_image[:, :, :3] * armor_alpha).astype(np.uint8)
    vis[:, :, 3] = np.maximum(vis[:, :, 3], armor_image[:, :, 3])

    # Add color overlays
    # Green tint for covering
    vis[:, :, 1] = np.where(green, np.minimum(255, vis[:, :, 1].astype(np.int16) + 60).astype(np.uint8), vis[:, :, 1])

    # Red tint for floating
    vis[:, :, 2] = np.where(red, np.minimum(255, vis[:, :, 2].astype(np.int16) + 80).astype(np.uint8), vis[:, :, 2])

    # Blue highlight for uncovered
    vis[:, :, 0] = np.where(blue, np.minimum(255, vis[:, :, 0].astype(np.int16) + 120).astype(np.uint8), vis[:, :, 0])

    return vis, int(np.sum(blue)), int(np.sum(red))


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

        # Scale and align
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

        # BEFORE rotation: create visualization
        before_viz, before_blue, before_red = create_overlap_viz(base_img, armor, neck_y)
        before_viz = draw_skeleton(before_viz, scaled_kpts, (255, 255, 0))  # Yellow = armor skeleton
        before_viz = draw_skeleton(before_viz, base_kpts, (0, 255, 255))    # Cyan = base skeleton
        before_viz = add_label(before_viz, f"BEFORE: blue={before_blue} red={before_red}")

        # Apply rotations
        rotated_armor = armor.copy()
        rotated_kpts = scaled_kpts.copy()
        for chain in LIMB_CHAINS:
            rotated_armor, rotated_kpts = apply_chain_rotations(
                rotated_armor, rotated_kpts, base_kpts, chain
            )

        # AFTER rotation: create visualization
        after_viz, after_blue, after_red = create_overlap_viz(base_img, rotated_armor, neck_y)
        after_viz = draw_skeleton(after_viz, rotated_kpts, (255, 255, 0))  # Yellow = rotated armor skeleton
        after_viz = draw_skeleton(after_viz, base_kpts, (0, 255, 255))     # Cyan = base skeleton
        after_viz = add_label(after_viz, f"AFTER: blue={after_blue} red={after_red}")

        # Side by side
        comparison = np.hstack([before_viz, after_viz])

        cv2.imwrite(str(output_dir / f"{clothed_name}_rotation_debug.png"), comparison)
        print(f"Before: blue={before_blue}, red={before_red}")
        print(f"After:  blue={after_blue}, red={after_red}")
        print(f"Saved: {clothed_name}_rotation_debug.png")


if __name__ == "__main__":
    main()
