#!/usr/bin/env python3
"""Test soft-edge inpainting - eliminate hard edges before inpainting.

Problem: If we have [ARMOR_EDGE][GRAY], inpainting just the gray creates
a visible seam where new pixels meet the hard armor edge.

Solution: Include the hard armor edge in the inpaint mask, so the
inpainting recreates both the edge AND the gap as one smooth region.

Pipeline:
1. Find uncovered gray pixels
2. Find armor edge pixels adjacent to gray (the "hard edge")
3. Include both in inpaint mask
4. Inpaint - creates smooth transition instead of hard seam
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation, binary_erosion

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


def get_armor_edge_near_gray(
    armor_alpha: np.ndarray,
    uncovered_mask: np.ndarray,
    edge_width: int = 2
) -> np.ndarray:
    """Find armor edge pixels that are adjacent to uncovered gray pixels.

    These are the "hard edges" that would create visible seams if we
    just inpaint the gray without including the edge.
    """
    armor_mask = armor_alpha > 128

    # Dilate the uncovered mask to find pixels near it
    dilated_uncovered = binary_dilation(uncovered_mask, iterations=edge_width)

    # Armor edge near gray = armor pixels that are within edge_width of uncovered
    armor_edge_near_gray = armor_mask & dilated_uncovered

    # But exclude the interior of armor (only want the edge)
    # Edge = armor pixels that have at least one non-armor neighbor
    eroded_armor = binary_erosion(armor_mask, iterations=1)
    armor_edge = armor_mask & ~eroded_armor

    # Final: armor edge pixels that are near uncovered gray
    return armor_edge_near_gray | (armor_edge & dilated_uncovered)


def soft_edge_texture_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    base_image: np.ndarray,
    neck_y: int,
    scaled_clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    edge_width: int = 2
) -> np.ndarray:
    """Inpaint with soft edges - include armor edge in inpaint region.

    1. Find uncovered gray pixels
    2. Find armor edge pixels adjacent to gray
    3. Remove those edge pixels from armor (make them transparent)
    4. Inpaint the combined region (gray + former edge)
    """
    result = armor.copy()

    uncovered = get_uncovered_mask(base_image, armor, neck_y)

    if not np.any(uncovered):
        return result

    # Find armor edge pixels near gray
    armor_edge = get_armor_edge_near_gray(armor[:, :, 3], uncovered, edge_width)

    print(f"    Uncovered pixels: {np.sum(uncovered)}")
    print(f"    Armor edge pixels to remove: {np.sum(armor_edge)}")

    # Remove edge pixels from armor (make transparent)
    # This creates a softer boundary for inpainting
    result_with_removed_edge = result.copy()
    result_with_removed_edge[:, :, 3] = np.where(armor_edge, 0, result[:, :, 3])

    # Combined inpaint region: original uncovered + removed edge
    inpaint_region = uncovered | armor_edge

    print(f"    Total inpaint region: {np.sum(inpaint_region)}")

    # Now do texture borrow inpainting on the combined region
    if not np.any(inpaint_region):
        return result

    # TPS mapping for texture borrowing
    corners = np.array([[0, 0], [511, 0], [0, 511], [511, 511]], dtype=np.float64)
    src_all = np.vstack([scaled_clothed_kpts, corners])
    dst_all = np.vstack([base_kpts, corners])

    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0)

    inpaint_ys, inpaint_xs = np.where(inpaint_region)
    dst_coords = np.column_stack([inpaint_xs, inpaint_ys])
    src_xs = rbf_x(dst_coords)
    src_ys = rbf_y(dst_coords)

    h, w = armor.shape[:2]

    # For the removed edge pixels, we want to sample from the original armor
    # For the uncovered pixels, we sample from the clothed frame

    for i, (dst_y, dst_x) in enumerate(zip(inpaint_ys, inpaint_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        # Try to sample from original clothed frame at mapped position
        if 0 <= src_x < w and 0 <= src_y < h:
            if armor_mask[src_y, src_x] > 128:
                result_with_removed_edge[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result_with_removed_edge[dst_y, dst_x, 3] = 255
                continue

        # Fallback: find nearest armor pixel from the ORIGINAL armor (before edge removal)
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
                result_with_removed_edge[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                result_with_removed_edge[dst_y, dst_x, 3] = 255
                break

    return result_with_removed_edge


def regular_texture_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    uncovered_mask: np.ndarray,
    scaled_clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray
) -> np.ndarray:
    """Regular texture borrow inpaint (without soft edges) for comparison."""
    result = armor.copy()

    if not np.any(uncovered_mask):
        return result

    corners = np.array([[0, 0], [511, 0], [0, 511], [511, 511]], dtype=np.float64)
    src_all = np.vstack([scaled_clothed_kpts, corners])
    dst_all = np.vstack([base_kpts, corners])

    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0)

    uncovered_ys, uncovered_xs = np.where(uncovered_mask)
    dst_coords = np.column_stack([uncovered_xs, uncovered_ys])
    src_xs = rbf_x(dst_coords)
    src_ys = rbf_y(dst_coords)

    h, w = armor.shape[:2]

    for i, (dst_y, dst_x) in enumerate(zip(uncovered_ys, uncovered_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        if 0 <= src_x < w and 0 <= src_y < h:
            if armor_mask[src_y, src_x] > 128:
                result[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result[dst_y, dst_x, 3] = 255
                continue

        armor_alpha = armor[:, :, 3] > 128
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
        initial_uncovered = count_uncovered_pixels(base_img, armor, neck_y)
        print(f"Initial uncovered: {initial_uncovered}")

        # Regular texture inpaint (hard edge)
        print("\nRegular texture inpaint:")
        uncovered = get_uncovered_mask(base_img, armor, neck_y)
        regular_result = regular_texture_inpaint(
            armor, scaled_clothed, uncovered,
            scaled_kpts, base_kpts, scaled_armor_mask
        )

        # Soft edge texture inpaint
        print("\nSoft edge texture inpaint:")
        soft_result = soft_edge_texture_inpaint(
            armor, scaled_clothed, base_img, neck_y,
            scaled_kpts, base_kpts, scaled_armor_mask,
            edge_width=2
        )

        # Create comparison
        no_warp_comp = composite_on_base(base_img, armor)
        regular_comp = composite_on_base(base_img, regular_result)
        soft_comp = composite_on_base(base_img, soft_result)

        no_warp_l = add_label(no_warp_comp, f"no_warp:{initial_uncovered}")
        regular_l = add_label(regular_comp, "hard_edge")
        soft_l = add_label(soft_comp, "soft_edge")

        comparison = np.hstack([no_warp_l, regular_l, soft_l])

        cv2.imwrite(str(output_dir / f"{clothed_name}_soft_edge.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_soft_edge_only.png"), soft_comp)
        print(f"Saved: {clothed_name}_soft_edge.png")


if __name__ == "__main__":
    main()
