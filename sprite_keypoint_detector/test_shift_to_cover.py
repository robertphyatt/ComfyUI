#!/usr/bin/env python3
"""Shift armor to cover uncovered base, then inpaint remaining gaps.

Key insight: We care about reducing BLUE (uncovered base), not total pixel count.
- Moving armor inward covers blue areas with real armor
- This may create gaps on the outer edge, but those are over transparent areas
- Those outer gaps are safe to inpaint (they don't need to match base character)

Pipeline:
1. For each limb segment, find shift that minimizes uncovered base pixels
2. Apply shifts (even if they create gaps elsewhere)
3. Soft-edge inpaint remaining gaps
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


def get_uncovered_base_mask(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None
) -> np.ndarray:
    """Get mask of base character pixels NOT covered by armor (the blue areas)."""
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers

    if neck_y is not None:
        h = base_image.shape[0]
        valid_region = np.zeros((h, base_image.shape[1]), dtype=bool)
        valid_region[neck_y:, :] = True
        uncovered = uncovered & valid_region

    return uncovered


def count_uncovered_base(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: Optional[int] = None
) -> int:
    """Count pixels where base is visible but armor doesn't cover."""
    return int(np.sum(get_uncovered_base_mask(base_image, armor_image, neck_y)))


def create_segment_mask(
    armor_alpha: np.ndarray,
    joint_pos: np.ndarray,
    end_pos: np.ndarray,
    width: int = 25
) -> np.ndarray:
    h, w = armor_alpha.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(end_pos[0]), int(end_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)

    mask = mask & (armor_alpha > 128).astype(np.uint8) * 255
    return mask


def simulate_shift(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    dx: int,
    dy: int
) -> np.ndarray:
    """Simulate shifting a segment and return the resulting armor."""
    h, w = armor.shape[:2]
    result = armor.copy()

    # Extract segment
    segment_rgba = np.zeros_like(armor)
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


def find_best_shift_for_coverage(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    base_image: np.ndarray,
    neck_y: int,
    max_shift: int = 6,
    min_improvement: int = 10
) -> Tuple[int, int, int]:
    """Find shift that best reduces uncovered base pixels.

    Returns (dx, dy, improvement) where improvement is reduction in uncovered base.
    Only returns non-zero if improvement >= min_improvement.
    """
    current_uncovered = count_uncovered_base(base_image, armor, neck_y)

    best_shift = (0, 0)
    best_uncovered = current_uncovered

    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue

            shifted_armor = simulate_shift(armor, segment_mask, dx, dy)
            new_uncovered = count_uncovered_base(base_image, shifted_armor, neck_y)

            if new_uncovered < best_uncovered:
                best_uncovered = new_uncovered
                best_shift = (dx, dy)

    improvement = current_uncovered - best_uncovered

    if improvement >= min_improvement:
        return best_shift[0], best_shift[1], improvement
    else:
        return 0, 0, 0


def soft_edge_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    base_image: np.ndarray,
    neck_y: int,
    scaled_clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    edge_width: int = 2
) -> np.ndarray:
    """Inpaint uncovered areas with soft edges."""
    result = armor.copy()

    uncovered = get_uncovered_base_mask(base_image, armor, neck_y)
    if not np.any(uncovered):
        return result

    # Find armor edge near uncovered
    armor_visible = armor[:, :, 3] > 128
    dilated_uncovered = binary_dilation(uncovered, iterations=edge_width)
    eroded_armor = binary_erosion(armor_visible, iterations=1)
    armor_edge = armor_visible & ~eroded_armor
    edge_to_remove = armor_edge & dilated_uncovered

    # Combined inpaint region
    inpaint_region = uncovered | edge_to_remove

    if not np.any(inpaint_region):
        return result

    # Remove edge from result
    result[:, :, 3] = np.where(edge_to_remove, 0, result[:, :, 3])

    # TPS mapping
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

    for i, (dst_y, dst_x) in enumerate(zip(inpaint_ys, inpaint_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        if 0 <= src_x < w and 0 <= src_y < h:
            if armor_mask[src_y, src_x] > 128:
                result[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result[dst_y, dst_x, 3] = 255
                continue

        # Fallback
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
        initial_uncovered = count_uncovered_base(base_img, armor, neck_y)
        print(f"Initial uncovered base: {initial_uncovered}")

        # Phase 1: Shift segments to cover base
        print("\nPhase 1: Shifting segments to cover uncovered base...")
        shifted_armor = armor.copy()
        total_improvement = 0

        for joint_idx, end_idx, name in LIMB_SEGMENTS:
            joint_pos = base_kpts[joint_idx]
            end_pos = base_kpts[end_idx]

            segment_mask = create_segment_mask(shifted_armor[:, :, 3], joint_pos, end_pos)
            if not np.any(segment_mask):
                continue

            # First, see what's the best we can do
            current = count_uncovered_base(base_img, shifted_armor, neck_y)
            best_dx, best_dy, best_uncovered = 0, 0, current

            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    test = simulate_shift(shifted_armor, segment_mask, dx, dy)
                    unc = count_uncovered_base(base_img, test, neck_y)
                    if unc < best_uncovered:
                        best_uncovered = unc
                        best_dx, best_dy = dx, dy

            improvement = current - best_uncovered

            if improvement > 0:
                print(f"  {name}: shift ({best_dx:+d}, {best_dy:+d}) reduces uncovered by {improvement}")
                shifted_armor = simulate_shift(shifted_armor, segment_mask, best_dx, best_dy)
                total_improvement += improvement
            else:
                print(f"  {name}: best shift=({best_dx:+d}, {best_dy:+d}) improvement={improvement}")

        after_shift = count_uncovered_base(base_img, shifted_armor, neck_y)
        print(f"\nAfter shifts: {after_shift} uncovered (reduced by {initial_uncovered - after_shift})")

        # Phase 2: Soft-edge inpaint remaining
        print("\nPhase 2: Soft-edge inpainting remaining gaps...")
        final_armor = soft_edge_inpaint(
            shifted_armor, scaled_clothed, base_img, neck_y,
            scaled_kpts, base_kpts, scaled_armor_mask
        )
        final_uncovered = count_uncovered_base(base_img, final_armor, neck_y)
        print(f"After inpaint: {final_uncovered} uncovered")

        # Also do inpaint-only for comparison (no shift)
        inpaint_only = soft_edge_inpaint(
            armor, scaled_clothed, base_img, neck_y,
            scaled_kpts, base_kpts, scaled_armor_mask
        )

        # Create comparison
        no_warp_comp = composite_on_base(base_img, armor)
        shifted_comp = composite_on_base(base_img, shifted_armor)
        final_comp = composite_on_base(base_img, final_armor)
        inpaint_only_comp = composite_on_base(base_img, inpaint_only)

        no_warp_l = add_label(no_warp_comp, f"original:{initial_uncovered}")
        shifted_l = add_label(shifted_comp, f"shifted:{after_shift}")
        final_l = add_label(final_comp, f"shift+inpaint:{final_uncovered}")
        inpaint_l = add_label(inpaint_only_comp, "inpaint_only:0")

        comparison = np.hstack([no_warp_l, shifted_l, final_l, inpaint_l])

        cv2.imwrite(str(output_dir / f"{clothed_name}_shift_cover.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_shift_cover_final.png"), final_comp)
        print(f"Saved: {clothed_name}_shift_cover.png")


if __name__ == "__main__":
    main()
