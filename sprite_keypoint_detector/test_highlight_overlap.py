#!/usr/bin/env python3
"""Visualize armor pixels that overlap transparent areas on the base layer.

This helps identify where armor is "floating" over nothing vs covering the base.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json

sys.path.insert(0, str(Path(__file__).parent))
from keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


FRAME_MAPPINGS = {
    "clothed_frame_00": "base_frame_23",
    "clothed_frame_01": "base_frame_24",
}


@dataclass
class OptimizerConfig:
    scale_factor: float = 1.057


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


def create_overlap_visualization(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int
) -> np.ndarray:
    """Create visualization showing:
    - Green: armor covering base (good)
    - Red: armor over transparent/nothing (floating)
    - Blue: base not covered by armor (gray showing through)
    """
    h, w = base_image.shape[:2]

    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128

    # Apply neck_y constraint
    valid_region = np.zeros((h, w), dtype=bool)
    valid_region[neck_y:, :] = True

    # Categories:
    # Green = armor AND base (armor properly covering)
    armor_covering_base = armor_visible & base_visible & valid_region

    # Red = armor AND NOT base (armor floating over nothing)
    armor_floating = armor_visible & ~base_visible & valid_region

    # Blue = base AND NOT armor (gray showing through)
    base_uncovered = base_visible & ~armor_visible & valid_region

    # Create visualization on top of composite
    vis = np.zeros((h, w, 4), dtype=np.uint8)

    # Start with base
    vis[:, :, :] = base_image[:, :, :]

    # Overlay armor
    armor_alpha = armor_image[:, :, 3:4] / 255.0
    vis[:, :, :3] = (vis[:, :, :3] * (1 - armor_alpha) +
                    armor_image[:, :, :3] * armor_alpha).astype(np.uint8)
    vis[:, :, 3] = np.maximum(vis[:, :, 3], armor_image[:, :, 3])

    # Now add colored overlays for the categories
    # Green tint for armor covering base
    vis[:, :, 1] = np.where(armor_covering_base,
                           np.minimum(255, vis[:, :, 1].astype(np.int16) + 50).astype(np.uint8),
                           vis[:, :, 1])

    # Red tint for armor floating
    vis[:, :, 2] = np.where(armor_floating,
                           np.minimum(255, vis[:, :, 2].astype(np.int16) + 80).astype(np.uint8),
                           vis[:, :, 2])

    # Blue highlight for uncovered base (this is the problem area)
    vis[:, :, 0] = np.where(base_uncovered,
                           np.minimum(255, vis[:, :, 0].astype(np.int16) + 100).astype(np.uint8),
                           vis[:, :, 0])

    return vis, armor_covering_base, armor_floating, base_uncovered


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    result = img.copy()
    cv2.putText(result, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255, 255), 1)
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

        scaled_clothed, scaled_kpts = scale_and_align_image(
            clothed_img, clothed_kpts, base_kpts, config.scale_factor
        )
        mask_rgba = np.zeros((*mask_img.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = mask_img
        mask_rgba[:, :, 3] = mask_img
        scaled_mask, _ = scale_and_align_image(
            mask_rgba, clothed_kpts, base_kpts, config.scale_factor
        )
        scaled_armor_mask = scaled_mask[:, :, 0]
        armor = apply_mask_to_image(scaled_clothed, scaled_armor_mask)

        neck_y = int(base_kpts[1, 1])

        # Create visualization
        vis, covering, floating, uncovered = create_overlap_visualization(
            base_img, armor, neck_y
        )

        print(f"  Armor covering base (green): {np.sum(covering)}")
        print(f"  Armor floating (red): {np.sum(floating)}")
        print(f"  Base uncovered (blue): {np.sum(uncovered)}")

        # Also create a legend/key image
        legend = np.zeros((60, 300, 4), dtype=np.uint8)
        legend[:, :, 3] = 255
        legend[:, :, :3] = 40  # dark gray background

        # Green box
        legend[10:25, 10:30, 1] = 200
        legend[10:25, 10:30, 3] = 255
        cv2.putText(legend, "Armor covering base", (35, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255,255), 1)

        # Red box
        legend[25:40, 10:30, 2] = 200
        legend[25:40, 10:30, 3] = 255
        cv2.putText(legend, "Armor floating (over transparent)", (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255,255), 1)

        # Blue box
        legend[40:55, 10:30, 0] = 200
        legend[40:55, 10:30, 3] = 255
        cv2.putText(legend, "Base uncovered (gray showing)", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255,255), 1)

        # Save
        cv2.imwrite(str(output_dir / f"{clothed_name}_overlap_vis.png"), vis)
        cv2.imwrite(str(output_dir / f"overlap_legend.png"), legend)
        print(f"Saved: {clothed_name}_overlap_vis.png")


if __name__ == "__main__":
    main()
