#!/usr/bin/env python3
"""Analyze WHERE uncovered pixels are located relative to segments."""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
import json

sys.path.insert(0, str(Path(__file__).parent))
from keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


FRAME_MAPPINGS = {
    "clothed_frame_00": "base_frame_23",
}


@dataclass
class OptimizerConfig:
    scale_factor: float = 1.057


LIMB_SEGMENTS = [
    (2, 4, "L_upper_arm"),
    (4, 6, "L_forearm"),
    (3, 5, "R_upper_arm"),
    (5, 7, "R_forearm"),
    (10, 12, "L_thigh"),
    (12, 14, "L_shin"),
    (11, 13, "R_thigh"),
    (13, 15, "R_shin"),
    # Add torso
    (1, 8, "torso"),
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


def get_uncovered_base_mask(base_image, armor_image, neck_y):
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers
    h = base_image.shape[0]
    valid_region = np.zeros((h, base_image.shape[1]), dtype=bool)
    valid_region[neck_y:, :] = True
    return uncovered & valid_region


def create_segment_region(h, w, joint_pos, end_pos, width=40):
    """Create region mask for a segment (wider for analysis)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(end_pos[0]), int(end_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)
    return mask > 0


def main():
    base_dir = Path(__file__).parent.parent / "training_data"
    frames_dir = base_dir / "frames"
    masks_dir = base_dir / "masks_corrected"
    annotations_path = base_dir / "annotations.json"
    output_dir = base_dir / "skeleton_comparison"

    annotations = load_annotations(annotations_path)
    config = OptimizerConfig()

    clothed_name = "clothed_frame_00"
    base_name = "base_frame_23"

    clothed_img = cv2.imread(str(frames_dir / f"{clothed_name}.png"), cv2.IMREAD_UNCHANGED)
    base_img = cv2.imread(str(frames_dir / f"{base_name}.png"), cv2.IMREAD_UNCHANGED)

    mask_idx = clothed_name.split("_")[-1]
    mask_img = cv2.imread(str(masks_dir / f"mask_{mask_idx}.png"), cv2.IMREAD_UNCHANGED)
    if len(mask_img.shape) == 3:
        mask_img = mask_img[:, :, 0]

    clothed_kpts = get_keypoints_array(annotations, clothed_name)
    base_kpts = get_keypoints_array(annotations, base_name)

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
    h, w = base_img.shape[:2]

    uncovered = get_uncovered_base_mask(base_img, armor, neck_y)
    total_uncovered = np.sum(uncovered)

    print(f"Total uncovered pixels: {total_uncovered}")
    print("\nUncovered pixels by body region:")

    accounted = np.zeros_like(uncovered)

    for joint_idx, end_idx, name in LIMB_SEGMENTS:
        joint_pos = base_kpts[joint_idx]
        end_pos = base_kpts[end_idx]

        region = create_segment_region(h, w, joint_pos, end_pos, width=40)
        in_region = uncovered & region & ~accounted

        count = np.sum(in_region)
        if count > 0:
            print(f"  {name}: {count} pixels")
            accounted |= in_region

    # Unaccounted
    unaccounted = uncovered & ~accounted
    print(f"  Other/joints: {np.sum(unaccounted)} pixels")

    # Create visualization showing uncovered by region
    vis = base_img.copy()
    armor_alpha = armor[:, :, 3:4] / 255.0
    vis[:, :, :3] = (vis[:, :, :3] * (1 - armor_alpha) + armor[:, :, :3] * armor_alpha).astype(np.uint8)
    vis[:, :, 3] = np.maximum(vis[:, :, 3], armor[:, :, 3])

    # Color uncovered pixels by region
    colors = {
        "L_upper_arm": (255, 0, 0),    # Blue
        "L_forearm": (255, 128, 0),
        "R_upper_arm": (0, 0, 255),    # Red
        "R_forearm": (0, 128, 255),
        "L_thigh": (0, 255, 0),        # Green
        "L_shin": (0, 255, 128),
        "R_thigh": (255, 0, 255),      # Magenta
        "R_shin": (255, 128, 255),
        "torso": (0, 255, 255),        # Yellow
    }

    accounted = np.zeros_like(uncovered)
    for joint_idx, end_idx, name in LIMB_SEGMENTS:
        joint_pos = base_kpts[joint_idx]
        end_pos = base_kpts[end_idx]
        region = create_segment_region(h, w, joint_pos, end_pos, width=40)
        in_region = uncovered & region & ~accounted

        if np.any(in_region) and name in colors:
            color = colors[name]
            vis[:, :, 0] = np.where(in_region, color[0], vis[:, :, 0])
            vis[:, :, 1] = np.where(in_region, color[1], vis[:, :, 1])
            vis[:, :, 2] = np.where(in_region, color[2], vis[:, :, 2])
            vis[:, :, 3] = np.where(in_region, 255, vis[:, :, 3])

        accounted |= in_region

    # White for unaccounted
    unaccounted = uncovered & ~accounted
    vis[:, :, 0] = np.where(unaccounted, 255, vis[:, :, 0])
    vis[:, :, 1] = np.where(unaccounted, 255, vis[:, :, 1])
    vis[:, :, 2] = np.where(unaccounted, 255, vis[:, :, 2])

    cv2.imwrite(str(output_dir / f"{clothed_name}_uncovered_regions.png"), vis)
    print(f"\nSaved: {clothed_name}_uncovered_regions.png")


if __name__ == "__main__":
    main()
