#!/usr/bin/env python3
"""Test the full pipeline: shift segments first, then inpaint remaining gaps.

Pipeline:
1. Scale/align armor to base pose
2. Shift limb segments at joints to cover gray pixels (uses real armor texture)
3. Inpaint any remaining gaps using either:
   a) texture_borrow - samples from original clothed frame
   b) opencv - simple color diffusion

This combines the best of both: real armor pixels from shifts, then smart fill for leftovers.
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


# Define limb segments: (joint_idx, end_idx, name)
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
    width: int = 30
) -> np.ndarray:
    """Create a mask for a limb segment between two keypoints."""
    h, w = armor_alpha.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(end_pos[0]), int(end_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)

    mask = mask & (armor_alpha > 128).astype(np.uint8) * 255
    return mask


def find_shift_for_segment(
    uncovered_mask: np.ndarray,
    segment_mask: np.ndarray,
    joint_pos: np.ndarray,
    max_shift: int = 8
) -> Tuple[int, int]:
    """Find the best shift to cover uncovered pixels near this segment.

    Only returns a shift if net coverage improves (pixels covered > pixels exposed).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    segment_region = cv2.dilate(segment_mask, kernel, iterations=1)
    nearby_uncovered = uncovered_mask & (segment_region > 0)

    if not np.any(nearby_uncovered):
        return (0, 0)

    ys, xs = np.where(nearby_uncovered)
    uncovered_center = np.array([np.mean(xs), np.mean(ys)])

    seg_ys, seg_xs = np.where(segment_mask > 0)
    if len(seg_xs) == 0:
        return (0, 0)
    segment_center = np.array([np.mean(seg_xs), np.mean(seg_ys)])

    direction = uncovered_center - segment_center
    dist = np.linalg.norm(direction)

    if dist < 1:
        return (0, 0)

    direction = direction / dist

    best_shift = (0, 0)
    best_net_gain = 0  # Must have positive net gain to shift

    for magnitude in range(1, max_shift + 1):
        dx = int(round(direction[0] * magnitude))
        dy = int(round(direction[1] * magnitude))

        shifted_mask = np.roll(np.roll(segment_mask, dx, axis=1), dy, axis=0)

        # Pixels we'd cover (uncovered that shifted mask now covers)
        pixels_covered = np.sum(nearby_uncovered & (shifted_mask > 0))

        # Pixels we'd expose (original segment position that shifted doesn't cover)
        # Only count if those exposed pixels were covering uncovered areas
        original_pos = segment_mask > 0
        shifted_pos = shifted_mask > 0
        newly_exposed = original_pos & ~shifted_pos
        # We only care about exposing pixels that were covering the base
        # (not pixels that were already transparent)
        pixels_exposed = np.sum(newly_exposed)

        net_gain = pixels_covered - pixels_exposed

        if net_gain > best_net_gain:
            best_net_gain = net_gain
            best_shift = (dx, dy)

    return best_shift


def shift_segment(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    dx: int,
    dy: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift a segment of the armor image."""
    h, w = armor.shape[:2]
    result = armor.copy()

    segment_rgba = armor.copy()
    for c in range(4):
        segment_rgba[:, :, c] = np.where(segment_mask > 0, armor[:, :, c], 0)

    for c in range(4):
        result[:, :, c] = np.where(segment_mask > 0, 0, result[:, :, c])

    shifted_segment = np.zeros_like(segment_rgba)

    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)

    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)

    shifted_segment[dst_y1:dst_y2, dst_x1:dst_x2] = segment_rgba[src_y1:src_y2, src_x1:src_x2]

    shifted_alpha = shifted_segment[:, :, 3:4] / 255.0

    for c in range(3):
        result[:, :, c] = (shifted_segment[:, :, c] * shifted_alpha[:, :, 0] +
                          result[:, :, c] * (1 - shifted_alpha[:, :, 0])).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], shifted_segment[:, :, 3])

    original_edge = segment_mask > 0
    shifted_mask = np.roll(np.roll(segment_mask, dx, axis=1), dy, axis=0)
    edge_mask = original_edge & ~(shifted_mask > 0)

    return result, edge_mask.astype(np.uint8) * 255


def apply_shifts(
    armor: np.ndarray,
    base_image: np.ndarray,
    base_keypoints: np.ndarray,
    neck_y: int,
    max_shift: int = 6,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply segment shifts to cover gaps. Returns shifted armor and edge mask."""
    result = armor.copy()
    all_edges = np.zeros(armor.shape[:2], dtype=np.uint8)

    for joint_idx, end_idx, name in LIMB_SEGMENTS:
        joint_pos = base_keypoints[joint_idx]
        end_pos = base_keypoints[end_idx]

        segment_mask = create_segment_mask(result[:, :, 3], joint_pos, end_pos, width=25)

        if not np.any(segment_mask):
            continue

        uncovered = get_uncovered_mask(base_image, result, neck_y)
        dx, dy = find_shift_for_segment(uncovered, segment_mask, joint_pos, max_shift)

        if dx == 0 and dy == 0:
            continue

        if verbose:
            print(f"    {name}: shift ({dx:+d}, {dy:+d})")

        result, edge_mask = shift_segment(result, segment_mask, dx, dy)
        all_edges = np.maximum(all_edges, edge_mask)

    return result, all_edges


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

    # Add corners for TPS
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

        # Step 1: Apply shifts
        print("  Applying segment shifts...")
        shifted_armor, edge_mask = apply_shifts(armor, base_img, base_kpts, neck_y, max_shift=6)

        after_shift = count_uncovered_pixels(base_img, shifted_armor, neck_y)
        print(f"After shifts: {after_shift}")

        # Step 2a: Texture borrow inpaint on shifted result
        print("  Applying texture_borrow inpaint...")
        remaining_uncovered = get_uncovered_mask(base_img, shifted_armor, neck_y)
        shift_texture = texture_borrow_inpaint(
            shifted_armor, scaled_clothed, remaining_uncovered,
            scaled_kpts, base_kpts, scaled_armor_mask
        )
        shift_texture_final = count_uncovered_pixels(base_img, shift_texture, neck_y)
        print(f"Shift + texture_borrow: {shift_texture_final}")

        # Step 2b: OpenCV inpaint on shifted result
        print("  Applying opencv inpaint...")
        shift_opencv = opencv_inpaint(shifted_armor, base_img, remaining_uncovered)
        shift_opencv_final = count_uncovered_pixels(base_img, shift_opencv, neck_y)
        print(f"Shift + opencv: {shift_opencv_final}")

        # Also do inpaint-only (no shift) for comparison
        print("  Inpaint-only comparisons...")
        no_shift_uncovered = get_uncovered_mask(base_img, armor, neck_y)

        texture_only = texture_borrow_inpaint(
            armor, scaled_clothed, no_shift_uncovered,
            scaled_kpts, base_kpts, scaled_armor_mask
        )
        texture_only_final = count_uncovered_pixels(base_img, texture_only, neck_y)

        opencv_only = opencv_inpaint(armor, base_img, no_shift_uncovered)
        opencv_only_final = count_uncovered_pixels(base_img, opencv_only, neck_y)

        # Create 4-way comparison: no_warp, shift+texture, shift+opencv, texture_only
        no_warp_comp = composite_on_base(base_img, armor)
        shift_tex_comp = composite_on_base(base_img, shift_texture)
        shift_cv_comp = composite_on_base(base_img, shift_opencv)
        tex_only_comp = composite_on_base(base_img, texture_only)

        no_warp_labeled = add_label(no_warp_comp, f"no_warp:{initial_uncovered}")
        shift_tex_labeled = add_label(shift_tex_comp, f"shift+tex:{shift_texture_final}")
        shift_cv_labeled = add_label(shift_cv_comp, f"shift+cv:{shift_opencv_final}")
        tex_only_labeled = add_label(tex_only_comp, f"tex_only:{texture_only_final}")

        comparison = np.hstack([no_warp_labeled, shift_tex_labeled, shift_cv_labeled, tex_only_labeled])

        cv2.imwrite(str(output_dir / f"{clothed_name}_full_compare.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_shift_texture_only.png"), shift_tex_comp)
        print(f"Saved: {clothed_name}_full_compare.png")


if __name__ == "__main__":
    main()
