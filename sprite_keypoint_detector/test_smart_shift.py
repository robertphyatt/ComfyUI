#!/usr/bin/env python3
"""Test smart limb shifting - only shift if it actually improves coverage.

The key insight: we should only move a limb if doing so covers MORE gray pixels
than it exposes. If moving doesn't help, skip it and let inpainting handle it.

This avoids making limbs thicker when inpainting fills gaps - instead we
genuinely reposition the armor to better cover the base.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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


# Limb segments: (joint_idx, end_idx, name)
LIMB_SEGMENTS = [
    (2, 4, "left_upper_arm"),
    (4, 6, "left_forearm"),
    (3, 5, "right_upper_arm"),
    (5, 7, "right_forearm"),
    (10, 12, "left_thigh"),
    (12, 14, "left_shin"),
    (11, 13, "right_thigh"),
    (13, 15, "right_shin"),
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


def get_uncovered_mask(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None
) -> np.ndarray:
    """Get mask of base character pixels not covered by armor."""
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


def create_segment_mask(
    armor_alpha: np.ndarray,
    joint_pos: np.ndarray,
    end_pos: np.ndarray,
    width: int = 25
) -> np.ndarray:
    """Create mask for a limb segment."""
    h, w = armor_alpha.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(end_pos[0]), int(end_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)

    # Only include actual armor pixels
    mask = mask & (armor_alpha > 128).astype(np.uint8) * 255
    return mask


def evaluate_shift(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    base_image: np.ndarray,
    neck_y: int,
    dx: int,
    dy: int
) -> Tuple[int, int, int]:
    """Evaluate what happens if we shift this segment by (dx, dy).

    Returns: (gray_before, gray_after, net_improvement)
    - gray_before: uncovered pixels before shift
    - gray_after: uncovered pixels after shift
    - net_improvement: gray_before - gray_after (positive = better)
    """
    h, w = armor.shape[:2]

    # Current uncovered count
    gray_before = count_uncovered_pixels(base_image, armor, neck_y)

    # Simulate the shift
    test_armor = armor.copy()

    # Extract segment
    segment_rgba = armor.copy()
    for c in range(4):
        segment_rgba[:, :, c] = np.where(segment_mask > 0, armor[:, :, c], 0)

    # Remove segment from test armor
    for c in range(4):
        test_armor[:, :, c] = np.where(segment_mask > 0, 0, test_armor[:, :, c])

    # Shift segment
    shifted = np.zeros_like(segment_rgba)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = segment_rgba[src_y1:src_y2, src_x1:src_x2]

    # Composite shifted segment onto test armor
    shifted_alpha = shifted[:, :, 3:4] / 255.0
    for c in range(3):
        test_armor[:, :, c] = (shifted[:, :, c] * shifted_alpha[:, :, 0] +
                               test_armor[:, :, c] * (1 - shifted_alpha[:, :, 0])).astype(np.uint8)
    test_armor[:, :, 3] = np.maximum(test_armor[:, :, 3], shifted[:, :, 3])

    # Count uncovered after shift
    gray_after = count_uncovered_pixels(base_image, test_armor, neck_y)

    return gray_before, gray_after, gray_before - gray_after


def find_best_shift(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    base_image: np.ndarray,
    neck_y: int,
    max_shift: int = 8
) -> Tuple[int, int, int]:
    """Find the best shift for a segment that maximizes coverage improvement.

    Returns: (dx, dy, improvement)
    Only returns non-zero shift if improvement > 0.
    """
    best_shift = (0, 0)
    best_improvement = 0

    # Try shifts in all directions
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue

            _, _, improvement = evaluate_shift(
                armor, segment_mask, base_image, neck_y, dx, dy
            )

            if improvement > best_improvement:
                best_improvement = improvement
                best_shift = (dx, dy)

    return best_shift[0], best_shift[1], best_improvement


def apply_shift(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    dx: int,
    dy: int
) -> np.ndarray:
    """Apply a shift to a segment."""
    h, w = armor.shape[:2]
    result = armor.copy()

    # Extract segment
    segment_rgba = armor.copy()
    for c in range(4):
        segment_rgba[:, :, c] = np.where(segment_mask > 0, armor[:, :, c], 0)

    # Remove segment from result
    for c in range(4):
        result[:, :, c] = np.where(segment_mask > 0, 0, result[:, :, c])

    # Shift segment
    shifted = np.zeros_like(segment_rgba)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = segment_rgba[src_y1:src_y2, src_x1:src_x2]

    # Composite
    shifted_alpha = shifted[:, :, 3:4] / 255.0
    for c in range(3):
        result[:, :, c] = (shifted[:, :, c] * shifted_alpha[:, :, 0] +
                          result[:, :, c] * (1 - shifted_alpha[:, :, 0])).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], shifted[:, :, 3])

    return result


def texture_borrow_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    uncovered_mask: np.ndarray,
    scaled_clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray
) -> np.ndarray:
    """Inpaint by sampling from original clothed frame."""
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

        initial_uncovered = count_uncovered_pixels(base_img, armor, neck_y)
        print(f"Initial uncovered: {initial_uncovered}")

        # Try smart shifts for each limb segment
        print("\nEvaluating smart shifts:")
        shifted_armor = armor.copy()
        total_improvement = 0

        for joint_idx, end_idx, name in LIMB_SEGMENTS:
            joint_pos = base_kpts[joint_idx]
            end_pos = base_kpts[end_idx]

            segment_mask = create_segment_mask(shifted_armor[:, :, 3], joint_pos, end_pos)

            if not np.any(segment_mask):
                continue

            dx, dy, improvement = find_best_shift(
                shifted_armor, segment_mask, base_img, neck_y, max_shift=6
            )

            if improvement > 0:
                print(f"  {name}: shift ({dx:+d}, {dy:+d}) improves by {improvement} pixels")
                shifted_armor = apply_shift(shifted_armor, segment_mask, dx, dy)
                total_improvement += improvement
            else:
                print(f"  {name}: no beneficial shift found")

        after_shift = count_uncovered_pixels(base_img, shifted_armor, neck_y)
        print(f"\nAfter smart shifts: {after_shift} (improved by {initial_uncovered - after_shift})")

        # Now inpaint remaining gaps
        remaining_uncovered = get_uncovered_mask(base_img, shifted_armor, neck_y)
        final_armor = texture_borrow_inpaint(
            shifted_armor, scaled_clothed, remaining_uncovered,
            scaled_kpts, base_kpts, scaled_armor_mask
        )
        final_uncovered = count_uncovered_pixels(base_img, final_armor, neck_y)
        print(f"After inpaint: {final_uncovered}")

        # Also do inpaint-only for comparison
        inpaint_only = texture_borrow_inpaint(
            armor, scaled_clothed, get_uncovered_mask(base_img, armor, neck_y),
            scaled_kpts, base_kpts, scaled_armor_mask
        )

        # Create comparison
        no_warp_comp = composite_on_base(base_img, armor)
        shifted_comp = composite_on_base(base_img, shifted_armor)
        final_comp = composite_on_base(base_img, final_armor)
        inpaint_only_comp = composite_on_base(base_img, inpaint_only)

        no_warp_l = add_label(no_warp_comp, f"no_warp:{initial_uncovered}")
        shifted_l = add_label(shifted_comp, f"shifted:{after_shift}")
        final_l = add_label(final_comp, f"shift+inpaint:{final_uncovered}")
        inpaint_l = add_label(inpaint_only_comp, f"inpaint_only:0")

        comparison = np.hstack([no_warp_l, shifted_l, final_l, inpaint_l])

        cv2.imwrite(str(output_dir / f"{clothed_name}_smart_shift.png"), comparison)
        print(f"Saved: {clothed_name}_smart_shift.png")


if __name__ == "__main__":
    main()
