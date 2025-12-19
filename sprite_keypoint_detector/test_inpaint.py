#!/usr/bin/env python3
"""Test inpainting approach for filling uncovered armor gaps."""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent to path for imports when running standalone
sys.path.insert(0, str(Path(__file__).parent))

from keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS
import json
from scipy.interpolate import RBFInterpolator
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Inline necessary functions to avoid import issues
FRAME_MAPPINGS = {
    "clothed_frame_00": "base_frame_23",
    "clothed_frame_01": "base_frame_24",
}


@dataclass
class OptimizerConfig:
    scale_factor: float = 1.057
    optimize_indices: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15)
    step_size: int = 1
    max_iterations: int = 100


def load_annotations(annotations_path: Path) -> Dict[str, Dict]:
    with open(annotations_path) as f:
        return json.load(f)


def get_keypoints_array(annotations: Dict, frame_name: str) -> np.ndarray:
    key = f"{frame_name}.png"
    if key not in annotations:
        raise KeyError(f"Frame {key} not found in annotations")

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


def count_uncovered_pixels(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None
) -> int:
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers

    if neck_y is not None:
        h = base_image.shape[0]
        valid_region = np.zeros((h, base_image.shape[1]), dtype=bool)
        valid_region[neck_y:, :] = True
        uncovered = uncovered & valid_region

    return int(np.sum(uncovered))


def get_uncovered_mask(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None
) -> np.ndarray:
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers

    if neck_y is not None:
        h = base_image.shape[0]
        valid_region = np.zeros((h, base_image.shape[1]), dtype=bool)
        valid_region[neck_y:, :] = True
        uncovered = uncovered & valid_region

    return uncovered


def inpaint_uncovered_areas(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None,
    inpaint_radius: int = 3,
    dilation_radius: int = 2
) -> np.ndarray:
    uncovered = get_uncovered_mask(base_image, armor_image, neck_y)

    if not np.any(uncovered):
        return armor_image

    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        uncovered_dilated = cv2.dilate(uncovered.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        uncovered_dilated = uncovered

    inpaint_mask = (uncovered_dilated.astype(np.uint8)) * 255

    composite = base_image.copy()
    armor_alpha = armor_image[:, :, 3:4] / 255.0
    composite[:, :, :3] = (composite[:, :, :3] * (1 - armor_alpha) +
                          armor_image[:, :, :3] * armor_alpha).astype(np.uint8)

    composite_bgr = composite[:, :, :3]
    inpainted_bgr = cv2.inpaint(composite_bgr, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)

    result = armor_image.copy()

    for c in range(3):
        result[:, :, c] = np.where(uncovered_dilated, inpainted_bgr[:, :, c], result[:, :, c])

    result[:, :, 3] = np.where(uncovered_dilated, 255, result[:, :, 3])

    return result


def inpaint_armor_gaps(
    clothed_image: np.ndarray,
    clothed_keypoints: np.ndarray,
    base_image: np.ndarray,
    base_keypoints: np.ndarray,
    mask_image: np.ndarray,
    scale_factor: float = 1.057,
    inpaint_radius: int = 3,
    dilation_radius: int = 2
) -> Tuple[np.ndarray, int, int]:
    scaled_clothed, scaled_kpts = scale_and_align_image(
        clothed_image,
        clothed_keypoints,
        base_keypoints,
        scale_factor
    )

    mask_rgba = np.zeros((*mask_image.shape[:2], 4), dtype=np.uint8)
    if len(mask_image.shape) == 2:
        mask_rgba[:, :, 0] = mask_image
        mask_rgba[:, :, 3] = mask_image
    else:
        mask_rgba[:, :, 0] = mask_image[:, :, 0]
        mask_rgba[:, :, 3] = mask_image[:, :, 0]

    scaled_mask, _ = scale_and_align_image(
        mask_rgba,
        clothed_keypoints,
        base_keypoints,
        scale_factor
    )

    armor = apply_mask_to_image(scaled_clothed, scaled_mask[:, :, 0])
    neck_y = int(base_keypoints[1, 1])
    uncovered_before = count_uncovered_pixels(base_image, armor, neck_y=neck_y)

    inpainted = inpaint_uncovered_areas(
        base_image, armor, neck_y=neck_y,
        inpaint_radius=inpaint_radius,
        dilation_radius=dilation_radius
    )

    uncovered_after = count_uncovered_pixels(base_image, inpainted, neck_y=neck_y)

    return inpainted, uncovered_before, uncovered_after


def composite_on_base(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Composite overlay on base using alpha blending."""
    result = base.copy()
    mask = overlay[:, :, 3:4] / 255.0
    result[:, :, :3] = (result[:, :, :3] * (1 - mask) + overlay[:, :, :3] * mask).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])
    return result


def add_label(img: np.ndarray, label: str, position: str = "top") -> np.ndarray:
    """Add label text to image."""
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255, 255)  # White

    if position == "top":
        y = 15
    else:
        y = img.shape[0] - 5

    cv2.putText(result, label, (5, y), font, font_scale, color, thickness)
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

        # Load images
        clothed_img = cv2.imread(str(frames_dir / f"{clothed_name}.png"), cv2.IMREAD_UNCHANGED)
        base_img = cv2.imread(str(frames_dir / f"{base_name}.png"), cv2.IMREAD_UNCHANGED)

        mask_idx = clothed_name.split("_")[-1]
        mask_img = cv2.imread(str(masks_dir / f"mask_{mask_idx}.png"), cv2.IMREAD_UNCHANGED)
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]

        clothed_kpts = get_keypoints_array(annotations, clothed_name)
        base_kpts = get_keypoints_array(annotations, base_name)

        # Generate NO-WARP baseline for comparison
        scaled_clothed, scaled_kpts = scale_and_align_image(
            clothed_img, clothed_kpts, base_kpts, config.scale_factor
        )
        mask_rgba = np.zeros((*mask_img.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = mask_img
        mask_rgba[:, :, 3] = mask_img
        scaled_mask, _ = scale_and_align_image(
            mask_rgba, clothed_kpts, base_kpts, config.scale_factor
        )
        no_warp_armor = apply_mask_to_image(scaled_clothed, scaled_mask[:, :, 0])

        neck_y = int(base_kpts[1, 1])
        no_warp_uncovered = count_uncovered_pixels(base_img, no_warp_armor, neck_y=neck_y)
        print(f"NO-WARP uncovered: {no_warp_uncovered}")

        # Generate INPAINT result
        inpainted, uncovered_before, uncovered_after = inpaint_armor_gaps(
            clothed_img, clothed_kpts, base_img, base_kpts, mask_img,
            scale_factor=config.scale_factor,
            inpaint_radius=5,  # Larger radius for smoother results
            dilation_radius=3  # Slightly larger dilation
        )
        print(f"INPAINT: {uncovered_before} -> {uncovered_after}")

        # Composite images
        no_warp_composite = composite_on_base(base_img, no_warp_armor)
        inpaint_composite = composite_on_base(base_img, inpainted)

        # Add labels
        no_warp_labeled = add_label(no_warp_composite, f"no_warp: {no_warp_uncovered}")
        inpaint_labeled = add_label(inpaint_composite, f"inpaint: {uncovered_after}")

        # Side-by-side comparison
        comparison = np.hstack([no_warp_labeled, inpaint_labeled])

        # Save images
        cv2.imwrite(str(output_dir / f"{clothed_name}_inpaint.png"), comparison)
        print(f"Saved: {clothed_name}_inpaint.png")

        # Also save individual images for closer inspection
        cv2.imwrite(str(output_dir / f"{clothed_name}_no_warp_only.png"), no_warp_composite)
        cv2.imwrite(str(output_dir / f"{clothed_name}_inpaint_only.png"), inpaint_composite)


if __name__ == "__main__":
    main()
