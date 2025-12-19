#!/usr/bin/env python3
"""Test edge expansion + inpaint pipeline.

Instead of shifting segments (which exposes gaps), we:
1. Identify uncovered pixels adjacent to armor
2. Expand armor edges into those pixels by copying nearby armor colors
3. Inpaint any remaining gaps that expansion couldn't reach

This keeps all original armor in place and only adds to edges.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation

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


def expand_armor_edges(
    armor: np.ndarray,
    base_image: np.ndarray,
    neck_y: int,
    max_iterations: int = 15
) -> np.ndarray:
    """Expand armor edges into uncovered areas by copying nearest armor pixels.

    Each iteration:
    1. Find armor edge pixels (armor pixels adjacent to non-armor)
    2. Find uncovered pixels adjacent to armor
    3. For each such uncovered pixel, copy color from nearest armor edge pixel
    4. Repeat until no more uncovered pixels can be reached

    This preserves all original armor and only adds new pixels at edges.
    """
    result = armor.copy()
    armor_alpha = result[:, :, 3] > 128

    for iteration in range(max_iterations):
        # Current uncovered
        uncovered = get_uncovered_mask(base_image, result, neck_y)

        if not np.any(uncovered):
            break

        # Dilate armor by 1 pixel
        dilated = binary_dilation(armor_alpha)

        # Pixels to fill = uncovered AND in dilated region AND not already armor
        to_fill = uncovered & dilated & ~armor_alpha

        if not np.any(to_fill):
            break

        # For each pixel to fill, find nearest armor pixel and copy its color
        fill_ys, fill_xs = np.where(to_fill)

        for fy, fx in zip(fill_ys, fill_xs):
            # Search in small neighborhood for nearest armor pixel
            best_dist = float('inf')
            best_color = None

            for radius in range(1, 5):
                y1, y2 = max(0, fy - radius), min(result.shape[0], fy + radius + 1)
                x1, x2 = max(0, fx - radius), min(result.shape[1], fx + radius + 1)

                for ny in range(y1, y2):
                    for nx in range(x1, x2):
                        if armor_alpha[ny, nx]:
                            dist = (ny - fy) ** 2 + (nx - fx) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_color = result[ny, nx, :3].copy()

                if best_color is not None:
                    break

            if best_color is not None:
                result[fy, fx, :3] = best_color
                result[fy, fx, 3] = 255

        # Update armor alpha for next iteration
        armor_alpha = result[:, :, 3] > 128

    return result


def texture_borrow_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    uncovered_mask: np.ndarray,
    scaled_clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray
) -> np.ndarray:
    """Inpaint by sampling from original clothed frame using TPS mapping."""
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

        # Fallback: nearest armor pixel
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


def opencv_inpaint(
    armor: np.ndarray,
    base_image: np.ndarray,
    uncovered_mask: np.ndarray,
    inpaint_radius: int = 5
) -> np.ndarray:
    """Inpaint using OpenCV TELEA algorithm."""
    result = armor.copy()

    if not np.any(uncovered_mask):
        return result

    inpaint_mask = uncovered_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    composite = base_image.copy()
    armor_alpha = armor[:, :, 3:4] / 255.0
    composite[:, :, :3] = (composite[:, :, :3] * (1 - armor_alpha) +
                          armor[:, :, :3] * armor_alpha).astype(np.uint8)

    inpainted_bgr = cv2.inpaint(composite[:, :, :3], inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)

    mask_bool = inpaint_mask > 0
    for c in range(3):
        result[:, :, c] = np.where(mask_bool, inpainted_bgr[:, :, c], result[:, :, c])
    result[:, :, 3] = np.where(mask_bool, 255, result[:, :, 3])

    return result


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

        # Load images
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
        scaled_armor_mask = scaled_mask[:, :, 0]
        armor = apply_mask_to_image(scaled_clothed, scaled_armor_mask)

        neck_y = int(base_kpts[1, 1])

        # Count initial uncovered
        initial_uncovered = count_uncovered_pixels(base_img, armor, neck_y)
        print(f"Initial uncovered: {initial_uncovered}")

        # Step 1: Expand armor edges
        print("  Expanding armor edges...")
        expanded = expand_armor_edges(armor, base_img, neck_y, max_iterations=20)
        after_expand = count_uncovered_pixels(base_img, expanded, neck_y)
        print(f"After edge expansion: {after_expand}")

        # Step 2a: Texture borrow on remaining
        print("  Texture borrow inpaint on remaining...")
        remaining = get_uncovered_mask(base_img, expanded, neck_y)
        expand_texture = texture_borrow_inpaint(
            expanded, scaled_clothed, remaining,
            scaled_kpts, base_kpts, scaled_armor_mask
        )
        expand_texture_final = count_uncovered_pixels(base_img, expand_texture, neck_y)
        print(f"Expand + texture: {expand_texture_final}")

        # Step 2b: OpenCV on remaining
        print("  OpenCV inpaint on remaining...")
        expand_opencv = opencv_inpaint(expanded, base_img, remaining)
        expand_opencv_final = count_uncovered_pixels(base_img, expand_opencv, neck_y)
        print(f"Expand + opencv: {expand_opencv_final}")

        # Also texture-only and opencv-only for comparison
        no_expand_uncovered = get_uncovered_mask(base_img, armor, neck_y)
        texture_only = texture_borrow_inpaint(
            armor, scaled_clothed, no_expand_uncovered,
            scaled_kpts, base_kpts, scaled_armor_mask
        )
        opencv_only = opencv_inpaint(armor, base_img, no_expand_uncovered)

        # 5-way comparison
        no_warp_comp = composite_on_base(base_img, armor)
        expand_only_comp = composite_on_base(base_img, expanded)
        expand_tex_comp = composite_on_base(base_img, expand_texture)
        expand_cv_comp = composite_on_base(base_img, expand_opencv)
        tex_only_comp = composite_on_base(base_img, texture_only)

        no_warp_l = add_label(no_warp_comp, f"no_warp:{initial_uncovered}")
        expand_only_l = add_label(expand_only_comp, f"expand:{after_expand}")
        expand_tex_l = add_label(expand_tex_comp, f"exp+tex:{expand_texture_final}")
        expand_cv_l = add_label(expand_cv_comp, f"exp+cv:{expand_opencv_final}")
        tex_only_l = add_label(tex_only_comp, f"tex_only:0")

        comparison = np.hstack([no_warp_l, expand_only_l, expand_tex_l, expand_cv_l, tex_only_l])

        cv2.imwrite(str(output_dir / f"{clothed_name}_expand_compare.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_expand_texture_only.png"), expand_tex_comp)
        print(f"Saved: {clothed_name}_expand_compare.png")


if __name__ == "__main__":
    main()
