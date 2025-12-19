#!/usr/bin/env python3
"""Test texture borrowing - sample from original clothed frame.

The problem: uncovered areas exist because poses differ between clothed and base.
The solution: for uncovered pixels, map back to the original clothed frame and
sample the armor texture from there.

This uses the skeleton keypoints to establish correspondence between frames.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy.interpolate import RBFInterpolator

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


def count_uncovered_pixels(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None
) -> int:
    return int(np.sum(get_uncovered_mask(base_image, armor_image, neck_y)))


def create_coordinate_mapping(
    src_keypoints: np.ndarray,
    dst_keypoints: np.ndarray,
    image_size: int = 512
) -> Tuple[RBFInterpolator, RBFInterpolator]:
    """Create TPS mapping from dst (base) coordinates to src (clothed) coordinates.

    This lets us ask: "for this pixel in the base pose, where was it in the clothed pose?"
    """
    # Add corner anchors
    corners = np.array([[0, 0], [image_size-1, 0], [0, image_size-1], [image_size-1, image_size-1]], dtype=np.float64)

    src_all = np.vstack([src_keypoints, corners])
    dst_all = np.vstack([dst_keypoints, corners])

    # Map from dst to src (we want to look up source coordinates for destination pixels)
    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0)

    return rbf_x, rbf_y


def texture_borrow_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    uncovered_mask: np.ndarray,
    scaled_clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray
) -> np.ndarray:
    """Inpaint uncovered areas by sampling from original clothed frame.

    For each uncovered pixel:
    1. Map its coordinates from base pose to clothed pose using TPS
    2. Sample the color from the original clothed frame at those coordinates
    3. Only use if that location has armor (not base character)

    Args:
        armor: Current armor image (has gaps)
        original_clothed: Original clothed frame (scaled/aligned, full frame not just armor)
        uncovered_mask: Boolean mask of pixels to fill
        scaled_clothed_kpts: Keypoints of scaled/aligned clothed frame
        base_kpts: Keypoints of base frame
        armor_mask: Mask showing where armor is in clothed frame

    Returns:
        Armor with uncovered areas filled from clothed frame texture
    """
    result = armor.copy()

    if not np.any(uncovered_mask):
        return result

    # Create coordinate mapping: base -> clothed
    rbf_x, rbf_y = create_coordinate_mapping(scaled_clothed_kpts, base_kpts)

    # Get uncovered pixel coordinates
    uncovered_ys, uncovered_xs = np.where(uncovered_mask)

    # Map all uncovered pixels at once for efficiency
    dst_coords = np.column_stack([uncovered_xs, uncovered_ys])
    src_xs = rbf_x(dst_coords)
    src_ys = rbf_y(dst_coords)

    h, w = armor.shape[:2]

    filled_count = 0
    fallback_count = 0

    for i, (dst_y, dst_x) in enumerate(zip(uncovered_ys, uncovered_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        # Check bounds
        if 0 <= src_x < w and 0 <= src_y < h:
            # Check if source location has armor (not base character)
            if armor_mask[src_y, src_x] > 128:
                # Sample from original clothed frame
                result[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result[dst_y, dst_x, 3] = 255
                filled_count += 1
                continue

        # Fallback: find nearest armor pixel in current armor
        armor_alpha = armor[:, :, 3] > 128
        if np.any(armor_alpha):
            for radius in range(1, 30):
                y1, y2 = max(0, dst_y - radius), min(h, dst_y + radius + 1)
                x1, x2 = max(0, dst_x - radius), min(w, dst_x + radius + 1)
                box = armor_alpha[y1:y2, x1:x2]
                if np.any(box):
                    box_ys, box_xs = np.where(box)
                    abs_ys, abs_xs = box_ys + y1, box_xs + x1
                    distances = (abs_ys - dst_y) ** 2 + (abs_xs - dst_x) ** 2
                    closest = np.argmin(distances)
                    result[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                    result[dst_y, dst_x, 3] = 255
                    fallback_count += 1
                    break

    print(f"  Filled from texture: {filled_count}, fallback: {fallback_count}")
    return result


def composite_on_base(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    result = base.copy()
    mask = overlay[:, :, 3:4] / 255.0
    result[:, :, :3] = (result[:, :, :3] * (1 - mask) + overlay[:, :, :3] * mask).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])
    return result


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    result = img.copy()
    cv2.putText(result, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 255), 1)
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

        # Scale and align clothed frame
        scaled_clothed, scaled_kpts = scale_and_align_image(
            clothed_img, clothed_kpts, base_kpts, config.scale_factor
        )

        # Scale and align mask
        mask_rgba = np.zeros((*mask_img.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = mask_img
        mask_rgba[:, :, 3] = mask_img
        scaled_mask, _ = scale_and_align_image(
            mask_rgba, clothed_kpts, base_kpts, config.scale_factor
        )
        scaled_armor_mask = scaled_mask[:, :, 0]

        # Extract armor
        armor = apply_mask_to_image(scaled_clothed, scaled_armor_mask)

        neck_y = int(base_kpts[1, 1])

        # Get uncovered mask
        uncovered = get_uncovered_mask(base_img, armor, neck_y)
        uncovered_count = np.sum(uncovered)

        print(f"Uncovered pixels: {uncovered_count}")

        # Apply texture-borrowing inpaint
        print("Applying texture-borrow inpainting...")
        texture_inpainted = texture_borrow_inpaint(
            armor, scaled_clothed, uncovered,
            scaled_kpts, base_kpts, scaled_armor_mask
        )

        final_uncovered = count_uncovered_pixels(base_img, texture_inpainted, neck_y)
        print(f"After texture borrow: {final_uncovered}")

        # Also apply OpenCV inpaint for comparison
        print("Applying OpenCV inpaint for comparison...")
        inpaint_mask = uncovered.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)

        # Create composite for OpenCV inpainting
        composite_for_cv = base_img.copy()
        armor_alpha = armor[:, :, 3:4] / 255.0
        composite_for_cv[:, :, :3] = (composite_for_cv[:, :, :3] * (1 - armor_alpha) +
                                      armor[:, :, :3] * armor_alpha).astype(np.uint8)
        opencv_inpainted_bgr = cv2.inpaint(composite_for_cv[:, :, :3], inpaint_mask_dilated, 5, cv2.INPAINT_TELEA)

        opencv_result = armor.copy()
        opencv_mask = inpaint_mask_dilated > 0
        for c in range(3):
            opencv_result[:, :, c] = np.where(opencv_mask, opencv_inpainted_bgr[:, :, c], opencv_result[:, :, c])
        opencv_result[:, :, 3] = np.where(opencv_mask, 255, opencv_result[:, :, 3])

        opencv_uncovered = count_uncovered_pixels(base_img, opencv_result, neck_y)
        print(f"After OpenCV inpaint: {opencv_uncovered}")

        # Create comparison
        no_warp_composite = composite_on_base(base_img, armor)
        texture_composite = composite_on_base(base_img, texture_inpainted)
        opencv_composite = composite_on_base(base_img, opencv_result)

        no_warp_labeled = add_label(no_warp_composite, f"no_warp: {uncovered_count}")
        texture_labeled = add_label(texture_composite, f"texture_borrow: {final_uncovered}")
        opencv_labeled = add_label(opencv_composite, f"opencv: {opencv_uncovered}")

        comparison = np.hstack([no_warp_labeled, texture_labeled, opencv_labeled])

        cv2.imwrite(str(output_dir / f"{clothed_name}_texture_borrow.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_texture_borrow_only.png"), texture_composite)
        print(f"Saved: {clothed_name}_texture_borrow.png")


if __name__ == "__main__":
    main()
