#!/usr/bin/env python3
"""Full pipeline: Scale/align -> Rigid rotation -> Soft-edge inpaint -> Pixelize

Complete pipeline for clothing overlay:
1. Scale and align armor to base pose (neck-based)
2. Rigid rotations at joints to match base skeleton
3. Soft-edge texture inpainting to fill gaps
4. Pixelize (simple downscale/upscale) to clean up
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import json
import math
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).parent))
from keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


FRAME_MAPPINGS = {
    "clothed_frame_00": "base_frame_23",
    "clothed_frame_01": "base_frame_24",
}


@dataclass
class PipelineConfig:
    scale_factor: float = 1.057
    rotation_segment_width: int = 35
    edge_width: int = 2
    pixelize_factor: int = 3  # downscale then upscale by this factor


# Skeleton connections for rotation chains
LIMB_CHAINS = [
    [(2, 4, "L_upper_arm"), (4, 6, "L_forearm")],
    [(3, 5, "R_upper_arm"), (5, 7, "R_forearm")],
    [(10, 12, "L_thigh"), (12, 14, "L_shin")],
    [(11, 13, "R_thigh"), (13, 15, "R_shin")],
]

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7),
    (1, 8), (8, 10), (8, 11), (10, 12), (12, 14), (11, 13), (13, 15),
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


# ============ STEP 1: Scale and Align ============

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


# ============ STEP 2: Rigid Rotation ============

def get_bone_angle(joint: np.ndarray, child: np.ndarray) -> float:
    delta = child - joint
    return math.atan2(delta[1], delta[0])


def create_segment_mask(h: int, w: int, joint_pos: np.ndarray, child_pos: np.ndarray, width: int = 35) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(child_pos[0]), int(child_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)
    return mask > 0


def rotate_point(point: np.ndarray, pivot: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    p = point - pivot
    rotated = np.array([p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a])
    return rotated + pivot


def apply_chain_rotations(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    target_kpts: np.ndarray,
    chain: List[Tuple[int, int, str]],
    segment_width: int = 35
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = armor.shape[:2]
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    for joint_idx, child_idx, name in chain:
        current_angle = get_bone_angle(result_kpts[joint_idx], result_kpts[child_idx])
        target_angle = get_bone_angle(target_kpts[joint_idx], target_kpts[child_idx])
        delta_angle = target_angle - current_angle

        if abs(delta_angle) < 0.01:
            continue

        segment_mask = create_segment_mask(h, w, result_kpts[joint_idx], result_kpts[child_idx], width=segment_width)
        armor_in_region = (result[:, :, 3] > 128) & segment_mask

        if not np.any(armor_in_region):
            continue

        pivot = result_kpts[joint_idx]

        for c in range(4):
            result[:, :, c] = np.where(armor_in_region, 0, result[:, :, c])

        segment_img = armor.copy()
        for c in range(4):
            segment_img[:, :, c] = np.where(armor_in_region, armor[:, :, c], 0)

        angle_deg = math.degrees(delta_angle)
        M = cv2.getRotationMatrix2D((pivot[0], pivot[1]), -angle_deg, 1.0)

        rotated_segment = cv2.warpAffine(
            segment_img, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        rot_alpha = rotated_segment[:, :, 3:4] / 255.0
        for c in range(3):
            result[:, :, c] = (rotated_segment[:, :, c] * rot_alpha[:, :, 0] +
                              result[:, :, c] * (1 - rot_alpha[:, :, 0])).astype(np.uint8)
        result[:, :, 3] = np.maximum(result[:, :, 3], rotated_segment[:, :, 3])

        result_kpts[child_idx] = rotate_point(result_kpts[child_idx], pivot, delta_angle)
        armor = result.copy()

    return result, result_kpts


def apply_all_rotations(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    target_kpts: np.ndarray,
    config: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray]:
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    for chain in LIMB_CHAINS:
        result, result_kpts = apply_chain_rotations(
            result, result_kpts, target_kpts, chain, config.rotation_segment_width
        )

    return result, result_kpts


# ============ STEP 3: Soft-Edge Inpaint ============

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


def get_armor_edge_near_gray(
    armor_alpha: np.ndarray,
    uncovered_mask: np.ndarray,
    edge_width: int = 2
) -> np.ndarray:
    armor_mask = armor_alpha > 128
    dilated_uncovered = binary_dilation(uncovered_mask, iterations=edge_width)
    armor_edge_near_gray = armor_mask & dilated_uncovered
    eroded_armor = binary_erosion(armor_mask, iterations=1)
    armor_edge = armor_mask & ~eroded_armor
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
    result = armor.copy()

    uncovered = get_uncovered_mask(base_image, armor, neck_y)

    if not np.any(uncovered):
        return result

    armor_edge = get_armor_edge_near_gray(armor[:, :, 3], uncovered, edge_width)

    result_with_removed_edge = result.copy()
    result_with_removed_edge[:, :, 3] = np.where(armor_edge, 0, result[:, :, 3])

    inpaint_region = uncovered | armor_edge

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

    for i, (dst_y, dst_x) in enumerate(zip(inpaint_ys, inpaint_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        if 0 <= src_x < w and 0 <= src_y < h:
            if armor_mask[src_y, src_x] > 128:
                result_with_removed_edge[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result_with_removed_edge[dst_y, dst_x, 3] = 255
                continue

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


# ============ STEP 4: Pixelize ============

def pixelize(image: np.ndarray, factor: int = 4) -> np.ndarray:
    """Simple pixelize: downscale then upscale with nearest neighbor."""
    h, w = image.shape[:2]
    small_h, small_w = h // factor, w // factor

    # Downscale
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Upscale with nearest neighbor
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return result


# ============ Visualization Helpers ============

def composite_on_base(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    result = base.copy()
    mask = overlay[:, :, 3:4] / 255.0
    result[:, :, :3] = (result[:, :, :3] * (1 - mask) + overlay[:, :, :3] * mask).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])
    return result


def create_overlap_viz(base_image: np.ndarray, armor_image: np.ndarray, neck_y: int) -> Tuple[np.ndarray, int, int]:
    """Create blue/green/red visualization."""
    h, w = base_image.shape[:2]

    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128

    valid = np.zeros((h, w), dtype=bool)
    valid[neck_y:, :] = True

    green = armor_visible & base_visible & valid
    red = armor_visible & ~base_visible & valid
    blue = base_visible & ~armor_visible & valid

    vis = base_image.copy()
    armor_alpha = armor_image[:, :, 3:4] / 255.0
    vis[:, :, :3] = (vis[:, :, :3] * (1 - armor_alpha) + armor_image[:, :, :3] * armor_alpha).astype(np.uint8)
    vis[:, :, 3] = np.maximum(vis[:, :, 3], armor_image[:, :, 3])

    vis[:, :, 1] = np.where(green, np.minimum(255, vis[:, :, 1].astype(np.int16) + 60).astype(np.uint8), vis[:, :, 1])
    vis[:, :, 2] = np.where(red, np.minimum(255, vis[:, :, 2].astype(np.int16) + 80).astype(np.uint8), vis[:, :, 2])
    vis[:, :, 0] = np.where(blue, np.minimum(255, vis[:, :, 0].astype(np.int16) + 120).astype(np.uint8), vis[:, :, 0])

    return vis, int(np.sum(blue)), int(np.sum(red))


def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int], thickness: int = 2):
    result = img.copy()
    for i, j in SKELETON_CONNECTIONS:
        pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(result, pt1, pt2, color + (255,), thickness)
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(result, (int(x), int(y)), 3, color + (255,), -1)
    return result


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    result = img.copy()
    cv2.putText(result, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255, 255), 1)
    return result


def count_pixels(base_image: np.ndarray, armor_image: np.ndarray, neck_y: int) -> Tuple[int, int]:
    """Count blue (uncovered) and red (floating) pixels."""
    h, w = base_image.shape[:2]
    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128
    valid = np.zeros((h, w), dtype=bool)
    valid[neck_y:, :] = True
    blue = base_visible & ~armor_visible & valid
    red = armor_visible & ~base_visible & valid
    return int(np.sum(blue)), int(np.sum(red))


# ============ Main Pipeline ============

def run_full_pipeline(
    clothed_img: np.ndarray,
    base_img: np.ndarray,
    mask_img: np.ndarray,
    clothed_kpts: np.ndarray,
    base_kpts: np.ndarray,
    config: PipelineConfig
) -> Dict[str, np.ndarray]:
    """Run the full pipeline and return intermediate results."""
    results = {}

    # Step 1: Scale and align
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
    armor_aligned = apply_mask_to_image(scaled_clothed, scaled_armor_mask)

    results['aligned'] = armor_aligned.copy()
    results['aligned_kpts'] = scaled_kpts.copy()

    # Step 2: Rigid rotation
    armor_rotated, rotated_kpts = apply_all_rotations(
        armor_aligned, scaled_kpts, base_kpts, config
    )

    results['rotated'] = armor_rotated.copy()
    results['rotated_kpts'] = rotated_kpts.copy()

    # Step 3: Soft-edge inpaint
    neck_y = int(base_kpts[1, 1])
    armor_inpainted = soft_edge_texture_inpaint(
        armor_rotated, scaled_clothed, base_img, neck_y,
        scaled_kpts, base_kpts, scaled_armor_mask, config.edge_width
    )

    results['inpainted'] = armor_inpainted.copy()

    # Step 4: Pixelize
    armor_pixelized = pixelize(armor_inpainted, config.pixelize_factor)

    results['pixelized'] = armor_pixelized.copy()

    # Also store intermediate composites
    results['scaled_clothed'] = scaled_clothed
    results['scaled_armor_mask'] = scaled_armor_mask
    results['base_kpts'] = base_kpts
    results['neck_y'] = neck_y

    return results


def main():
    base_dir = Path(__file__).parent.parent / "training_data"
    frames_dir = base_dir / "frames"
    masks_dir = base_dir / "masks_corrected"
    annotations_path = base_dir / "annotations.json"
    output_dir = base_dir / "skeleton_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_path)
    config = PipelineConfig()

    for clothed_name, base_name in FRAME_MAPPINGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {clothed_name} -> {base_name}")
        print(f"{'='*60}")

        clothed_img = cv2.imread(str(frames_dir / f"{clothed_name}.png"), cv2.IMREAD_UNCHANGED)
        base_img = cv2.imread(str(frames_dir / f"{base_name}.png"), cv2.IMREAD_UNCHANGED)

        mask_idx = clothed_name.split("_")[-1]
        mask_img = cv2.imread(str(masks_dir / f"mask_{mask_idx}.png"), cv2.IMREAD_UNCHANGED)
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]

        clothed_kpts = get_keypoints_array(annotations, clothed_name)
        base_kpts = get_keypoints_array(annotations, base_name)

        # Run full pipeline (without pixelization - we'll do that separately)
        results = run_full_pipeline(
            clothed_img, base_img, mask_img, clothed_kpts, base_kpts, config
        )

        neck_y = results['neck_y']
        inpainted = results['inpainted']

        # Create pixelization comparison with factors 1-5
        pixelized_stages = []
        for factor in range(1, 6):
            if factor == 1:
                pix = inpainted.copy()  # No pixelization
            else:
                pix = pixelize(inpainted, factor)

            pix_blue, pix_red = count_pixels(base_img, pix, neck_y)
            pix_comp = composite_on_base(base_img, pix)
            pix_viz = add_label(pix_comp, f"factor={factor} b:{pix_blue}")
            pixelized_stages.append(pix_viz)
            print(f"Pixelize factor {factor}: blue={pix_blue}, red={pix_red}")

        # Create side-by-side comparison of all pixelization factors
        pixelize_comparison = np.hstack(pixelized_stages)
        cv2.imwrite(str(output_dir / f"{clothed_name}_pixelize_compare.png"), pixelize_comparison)
        print(f"\nSaved: {clothed_name}_pixelize_compare.png")


if __name__ == "__main__":
    main()
