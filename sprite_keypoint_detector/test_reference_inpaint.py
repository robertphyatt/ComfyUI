#!/usr/bin/env python3
"""Test reference-aware inpainting - borrow texture from original clothed frame.

Instead of OpenCV's color diffusion, we:
1. For each uncovered pixel, find the nearest armor pixel in the ORIGINAL clothed frame
2. Copy that pixel's color (which has real armor texture)
3. This preserves the actual armor appearance instead of just smoothing
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import ndimage

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


def reference_inpaint(
    armor: np.ndarray,
    reference_armor: np.ndarray,
    uncovered_mask: np.ndarray,
    max_search_radius: int = 30
) -> np.ndarray:
    """Inpaint uncovered areas by borrowing pixels from reference armor.

    For each uncovered pixel, find the nearest pixel in the reference armor
    (the original scaled/aligned armor before any shifts) and copy its color.

    Args:
        armor: Current armor image (after shifts, has gaps)
        reference_armor: Original armor image (before shifts, complete)
        uncovered_mask: Boolean mask of pixels to fill
        max_search_radius: Maximum distance to search for source pixels

    Returns:
        Armor with uncovered areas filled from reference
    """
    result = armor.copy()

    if not np.any(uncovered_mask):
        return result

    # Get reference armor alpha mask
    ref_alpha = reference_armor[:, :, 3] > 128

    # For each uncovered pixel, find nearest reference armor pixel
    # Use distance transform to find nearest armor pixel efficiently

    # Distance from each pixel to nearest reference armor pixel
    # We invert the mask: distance_transform gives distance to nearest 0
    dist_to_ref = ndimage.distance_transform_edt(~ref_alpha)

    # Get coordinates of uncovered pixels
    uncovered_ys, uncovered_xs = np.where(uncovered_mask)

    for y, x in zip(uncovered_ys, uncovered_xs):
        # Find nearest reference armor pixel within search radius
        # Search in expanding squares
        found = False

        for radius in range(1, max_search_radius + 1):
            # Define search box
            y1, y2 = max(0, y - radius), min(armor.shape[0], y + radius + 1)
            x1, x2 = max(0, x - radius), min(armor.shape[1], x + radius + 1)

            # Check for reference armor pixels in this box
            box_ref = ref_alpha[y1:y2, x1:x2]

            if np.any(box_ref):
                # Find the closest one
                box_ys, box_xs = np.where(box_ref)

                # Convert to absolute coordinates
                abs_ys = box_ys + y1
                abs_xs = box_xs + x1

                # Find closest
                distances = (abs_ys - y) ** 2 + (abs_xs - x) ** 2
                closest_idx = np.argmin(distances)

                src_y, src_x = abs_ys[closest_idx], abs_xs[closest_idx]

                # Copy pixel from reference
                result[y, x, :3] = reference_armor[src_y, src_x, :3]
                result[y, x, 3] = 255
                found = True
                break

        if not found:
            # Fallback: just use nearest armor pixel from current armor
            armor_alpha = armor[:, :, 3] > 128
            if np.any(armor_alpha):
                dist_to_armor = ndimage.distance_transform_edt(~armor_alpha)
                # Find nearest by checking neighborhood
                for radius in range(1, max_search_radius + 1):
                    y1, y2 = max(0, y - radius), min(armor.shape[0], y + radius + 1)
                    x1, x2 = max(0, x - radius), min(armor.shape[1], x + radius + 1)
                    box_armor = armor_alpha[y1:y2, x1:x2]
                    if np.any(box_armor):
                        box_ys, box_xs = np.where(box_armor)
                        abs_ys = box_ys + y1
                        abs_xs = box_xs + x1
                        distances = (abs_ys - y) ** 2 + (abs_xs - x) ** 2
                        closest_idx = np.argmin(distances)
                        src_y, src_x = abs_ys[closest_idx], abs_xs[closest_idx]
                        result[y, x, :3] = armor[src_y, src_x, :3]
                        result[y, x, 3] = 255
                        break

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

        # Scale and align to get the reference armor (before any shifts)
        scaled_clothed, scaled_kpts = scale_and_align_image(
            clothed_img, clothed_kpts, base_kpts, config.scale_factor
        )
        mask_rgba = np.zeros((*mask_img.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = mask_img
        mask_rgba[:, :, 3] = mask_img
        scaled_mask, _ = scale_and_align_image(
            mask_rgba, clothed_kpts, base_kpts, config.scale_factor
        )
        reference_armor = apply_mask_to_image(scaled_clothed, scaled_mask[:, :, 0])

        neck_y = int(base_kpts[1, 1])

        # Get uncovered mask
        uncovered = get_uncovered_mask(base_img, reference_armor, neck_y)
        uncovered_count = np.sum(uncovered)

        print(f"Uncovered pixels: {uncovered_count}")

        # Apply reference-aware inpainting
        print("Applying reference-aware inpainting...")
        inpainted = reference_inpaint(
            reference_armor, reference_armor, uncovered,
            max_search_radius=50
        )

        final_uncovered = count_uncovered_pixels(base_img, inpainted, neck_y)
        print(f"After reference inpaint: {final_uncovered}")

        # Also try OpenCV inpaint for comparison
        print("Applying OpenCV inpaint for comparison...")
        inpaint_mask = uncovered.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)

        composite = base_img.copy()
        ref_alpha = reference_armor[:, :, 3:4] / 255.0
        composite[:, :, :3] = (composite[:, :, :3] * (1 - ref_alpha) +
                              reference_armor[:, :, :3] * ref_alpha).astype(np.uint8)
        opencv_inpainted_bgr = cv2.inpaint(composite[:, :, :3], inpaint_mask_dilated, 5, cv2.INPAINT_TELEA)

        opencv_result = reference_armor.copy()
        opencv_mask = inpaint_mask_dilated > 0
        for c in range(3):
            opencv_result[:, :, c] = np.where(opencv_mask, opencv_inpainted_bgr[:, :, c], opencv_result[:, :, c])
        opencv_result[:, :, 3] = np.where(opencv_mask, 255, opencv_result[:, :, 3])

        opencv_final = count_uncovered_pixels(base_img, opencv_result, neck_y)
        print(f"After OpenCV inpaint: {opencv_final}")

        # Create comparison
        no_warp_composite = composite_on_base(base_img, reference_armor)
        ref_inpaint_composite = composite_on_base(base_img, inpainted)
        opencv_composite = composite_on_base(base_img, opencv_result)

        no_warp_labeled = add_label(no_warp_composite, f"no_warp: {uncovered_count}")
        ref_labeled = add_label(ref_inpaint_composite, f"ref_inpaint: {final_uncovered}")
        opencv_labeled = add_label(opencv_composite, f"opencv: {opencv_final}")

        comparison = np.hstack([no_warp_labeled, ref_labeled, opencv_labeled])

        cv2.imwrite(str(output_dir / f"{clothed_name}_ref_inpaint.png"), comparison)
        print(f"Saved: {clothed_name}_ref_inpaint.png")


if __name__ == "__main__":
    main()
