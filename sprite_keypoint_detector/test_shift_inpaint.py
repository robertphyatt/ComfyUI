#!/usr/bin/env python3
"""Test joint-shift + inpaint hybrid approach for filling armor gaps.

Strategy:
1. For each limb segment (upper arm, forearm, thigh, shin), detect if there are
   uncovered pixels near the joint
2. Calculate a small shift vector to move that segment's pixels over the gap
3. Apply the shift to just that segment (using a mask)
4. Mark shifted edges + any remaining uncovered areas for inpainting
5. Inpaint to blend seams and fill remaining gaps
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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


# Define limb segments: (joint_idx, end_idx, name)
# Each segment goes from a joint to the next point in the chain
LIMB_SEGMENTS = [
    # Arms
    (2, 4, "left_upper_arm"),    # left_shoulder -> left_elbow
    (4, 6, "left_forearm"),      # left_elbow -> left_wrist
    (3, 5, "right_upper_arm"),   # right_shoulder -> right_elbow
    (5, 7, "right_forearm"),     # right_elbow -> right_wrist
    # Legs
    (10, 12, "left_thigh"),      # left_hip -> left_knee
    (12, 14, "left_shin"),       # left_knee -> left_ankle
    (11, 13, "right_thigh"),     # right_hip -> right_knee
    (13, 15, "right_shin"),      # right_knee -> right_ankle
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
    """Create a mask for a limb segment between two keypoints.

    Uses a capsule shape (rectangle with rounded ends) along the bone.
    """
    h, w = armor_alpha.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw thick line between joint and end
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(end_pos[0]), int(end_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)

    # Add circles at endpoints for capsule shape
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)

    # Only include pixels that actually have armor
    mask = mask & (armor_alpha > 128).astype(np.uint8) * 255

    return mask


def find_shift_for_segment(
    uncovered_mask: np.ndarray,
    segment_mask: np.ndarray,
    joint_pos: np.ndarray,
    max_shift: int = 8
) -> Tuple[int, int]:
    """Find the best shift to cover uncovered pixels near this segment.

    Returns (dx, dy) shift vector.
    """
    # Find uncovered pixels that are near this segment
    # Dilate segment mask to find nearby uncovered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    segment_region = cv2.dilate(segment_mask, kernel, iterations=1)

    nearby_uncovered = uncovered_mask & (segment_region > 0)

    if not np.any(nearby_uncovered):
        return (0, 0)  # No uncovered pixels near this segment

    # Find centroid of nearby uncovered pixels
    ys, xs = np.where(nearby_uncovered)
    uncovered_center = np.array([np.mean(xs), np.mean(ys)])

    # Find centroid of segment armor pixels
    seg_ys, seg_xs = np.where(segment_mask > 0)
    if len(seg_xs) == 0:
        return (0, 0)
    segment_center = np.array([np.mean(seg_xs), np.mean(seg_ys)])

    # Shift direction: from segment center toward uncovered center
    direction = uncovered_center - segment_center
    dist = np.linalg.norm(direction)

    if dist < 1:
        return (0, 0)

    # Normalize and scale by a small amount (we want subtle shifts)
    direction = direction / dist

    # Try different shift magnitudes and pick best one
    best_shift = (0, 0)
    best_coverage = 0

    for magnitude in range(1, max_shift + 1):
        dx = int(round(direction[0] * magnitude))
        dy = int(round(direction[1] * magnitude))

        # Simulate shifting segment mask
        shifted_mask = np.roll(np.roll(segment_mask, dx, axis=1), dy, axis=0)

        # Count how many uncovered pixels this would cover
        coverage = np.sum(nearby_uncovered & (shifted_mask > 0))

        if coverage > best_coverage:
            best_coverage = coverage
            best_shift = (dx, dy)

    return best_shift


def shift_segment(
    armor: np.ndarray,
    segment_mask: np.ndarray,
    dx: int,
    dy: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift a segment of the armor image.

    Returns:
        (shifted_armor, edge_mask) where edge_mask marks areas needing inpainting
    """
    h, w = armor.shape[:2]
    result = armor.copy()

    # Extract segment pixels
    segment_rgba = armor.copy()
    for c in range(4):
        segment_rgba[:, :, c] = np.where(segment_mask > 0, armor[:, :, c], 0)

    # Clear segment from result (will be replaced with shifted version)
    for c in range(4):
        result[:, :, c] = np.where(segment_mask > 0, 0, result[:, :, c])

    # Shift segment
    shifted_segment = np.zeros_like(segment_rgba)

    # Calculate source and destination regions
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)

    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)

    shifted_segment[dst_y1:dst_y2, dst_x1:dst_x2] = segment_rgba[src_y1:src_y2, src_x1:src_x2]

    # Composite shifted segment onto result (shifted segment on top)
    shifted_alpha = shifted_segment[:, :, 3:4] / 255.0
    result_alpha = result[:, :, 3:4] / 255.0

    # Where shifted segment has content, use it
    for c in range(3):
        result[:, :, c] = (shifted_segment[:, :, c] * shifted_alpha[:, :, 0] +
                          result[:, :, c] * (1 - shifted_alpha[:, :, 0])).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], shifted_segment[:, :, 3])

    # Create edge mask - the boundary where segment was shifted from
    # This is where we'll need inpainting to blend
    original_edge = segment_mask > 0
    shifted_mask = np.roll(np.roll(segment_mask, dx, axis=1), dy, axis=0)

    # Edge is where original was but shifted isn't (the gap left behind)
    edge_mask = original_edge & ~(shifted_mask > 0)

    return result, edge_mask.astype(np.uint8) * 255


def shift_and_inpaint(
    armor: np.ndarray,
    base_image: np.ndarray,
    base_keypoints: np.ndarray,
    neck_y: int,
    max_shift: int = 6,
    inpaint_radius: int = 5,
    verbose: bool = True
) -> np.ndarray:
    """Apply segment shifts then inpaint remaining gaps.

    1. For each limb segment, find optimal shift to cover nearby gaps
    2. Apply shifts sequentially
    3. Inpaint all remaining uncovered areas + shift edges
    """
    result = armor.copy()
    all_edges = np.zeros(armor.shape[:2], dtype=np.uint8)

    armor_alpha = armor[:, :, 3]

    for joint_idx, end_idx, name in LIMB_SEGMENTS:
        joint_pos = base_keypoints[joint_idx]
        end_pos = base_keypoints[end_idx]

        # Create segment mask
        segment_mask = create_segment_mask(result[:, :, 3], joint_pos, end_pos, width=25)

        if not np.any(segment_mask):
            continue

        # Get current uncovered mask
        uncovered = get_uncovered_mask(base_image, result, neck_y)

        # Find best shift for this segment
        dx, dy = find_shift_for_segment(uncovered, segment_mask, joint_pos, max_shift)

        if dx == 0 and dy == 0:
            continue

        if verbose:
            print(f"  {name}: shift ({dx:+d}, {dy:+d})")

        # Apply shift
        result, edge_mask = shift_segment(result, segment_mask, dx, dy)
        all_edges = np.maximum(all_edges, edge_mask)

    # Now inpaint: uncovered areas + shift edges
    uncovered = get_uncovered_mask(base_image, result, neck_y)

    # Combine uncovered with edges for inpainting
    inpaint_mask = (uncovered.astype(np.uint8) * 255) | all_edges

    # Dilate slightly for better blending
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    if not np.any(inpaint_mask):
        return result

    # Create composite for inpainting
    composite = base_image.copy()
    result_alpha = result[:, :, 3:4] / 255.0
    composite[:, :, :3] = (composite[:, :, :3] * (1 - result_alpha) +
                          result[:, :, :3] * result_alpha).astype(np.uint8)

    # Inpaint
    inpainted_bgr = cv2.inpaint(composite[:, :, :3], inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Apply inpainted pixels back to result
    inpaint_bool = inpaint_mask > 0
    for c in range(3):
        result[:, :, c] = np.where(inpaint_bool, inpainted_bgr[:, :, c], result[:, :, c])
    result[:, :, 3] = np.where(inpaint_bool, 255, result[:, :, 3])

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
        armor = apply_mask_to_image(scaled_clothed, scaled_mask[:, :, 0])

        neck_y = int(base_kpts[1, 1])

        # Baseline: no modifications
        no_warp_uncovered = count_uncovered_pixels(base_img, armor, neck_y)
        print(f"NO-WARP uncovered: {no_warp_uncovered}")

        # Apply shift + inpaint
        print("Applying segment shifts...")
        shifted_inpainted = shift_and_inpaint(
            armor, base_img, base_kpts, neck_y,
            max_shift=6,
            inpaint_radius=5,
            verbose=True
        )

        final_uncovered = count_uncovered_pixels(base_img, shifted_inpainted, neck_y)
        print(f"SHIFT+INPAINT uncovered: {final_uncovered}")

        # Create comparison image
        no_warp_composite = composite_on_base(base_img, armor)
        final_composite = composite_on_base(base_img, shifted_inpainted)

        no_warp_labeled = add_label(no_warp_composite, f"no_warp: {no_warp_uncovered}")
        final_labeled = add_label(final_composite, f"shift+inpaint: {final_uncovered}")

        comparison = np.hstack([no_warp_labeled, final_labeled])

        cv2.imwrite(str(output_dir / f"{clothed_name}_shift_inpaint.png"), comparison)
        cv2.imwrite(str(output_dir / f"{clothed_name}_shift_inpaint_only.png"), final_composite)
        print(f"Saved: {clothed_name}_shift_inpaint.png")
        print(f"Saved: {clothed_name}_shift_inpaint_only.png")


if __name__ == "__main__":
    main()
