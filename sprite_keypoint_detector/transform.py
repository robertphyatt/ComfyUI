"""Transform pipeline: scale, align, rotate, inpaint, pixelize."""

import cv2
import numpy as np
import math
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation, binary_erosion

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


@dataclass
class TransformConfig:
    """Configuration for the transform pipeline."""
    scale_factor: float = 1.057
    rotation_segment_width: int = 35
    edge_width: int = 2
    pixelize_factor: int = 3
    canvas_size: int = 512
    skip_rotation: bool = False  # Skip rotation step entirely (for good fits)


# Limb chains for rotation (joint_idx, child_idx, name)
LIMB_CHAINS = [
    [(2, 4, "L_upper_arm"), (4, 6, "L_forearm")],   # Left arm
    [(3, 5, "R_upper_arm"), (5, 7, "R_forearm")],   # Right arm
    [(10, 12, "L_thigh"), (12, 14, "L_shin")],      # Left leg
    [(11, 13, "R_thigh"), (13, 15, "R_shin")],      # Right leg
]


def get_keypoints_array(keypoints: Dict) -> np.ndarray:
    """Convert keypoints dict to numpy array."""
    result = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    for i, name in enumerate(KEYPOINT_NAMES):
        if name in keypoints:
            kp = keypoints[name]
            if isinstance(kp, list):
                result[i] = kp
            elif isinstance(kp, dict):
                result[i] = [kp["x"], kp["y"]]
    return result


# ============ Step 1: Scale and Align (neck + hip mean) ============

def scale_and_align(
    image: np.ndarray,
    image_kpts: np.ndarray,
    target_kpts: np.ndarray,
    config: TransformConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale image and align using mean of neck and hip offsets.

    Args:
        image: Source RGBA image
        image_kpts: Keypoints for source image
        target_kpts: Target keypoints to align to
        config: Transform configuration

    Returns:
        (aligned_image, aligned_keypoints)
    """
    h, w = image.shape[:2]
    scale = config.scale_factor
    canvas_size = config.canvas_size

    # Scale
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_kpts = image_kpts * scale

    # Compute mean offset from neck (idx 1) and mid-hip
    # Mid-hip is mean of left_hip (idx 10) and right_hip (idx 11)
    neck_idx = 1
    left_hip_idx = 10
    right_hip_idx = 11

    scaled_mid_hip = (scaled_kpts[left_hip_idx] + scaled_kpts[right_hip_idx]) / 2
    target_mid_hip = (target_kpts[left_hip_idx] + target_kpts[right_hip_idx]) / 2

    neck_offset = target_kpts[neck_idx] - scaled_kpts[neck_idx]
    hip_offset = target_mid_hip - scaled_mid_hip

    # Mean offset
    mean_offset = (neck_offset + hip_offset) / 2
    offset_x = int(round(mean_offset[0]))
    offset_y = int(round(mean_offset[1]))

    # Create canvas and place scaled image
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    src_x1 = max(0, -offset_x)
    src_x2 = min(new_w, canvas_size - offset_x)
    src_y1 = max(0, -offset_y)
    src_y2 = min(new_h, canvas_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = max(0, offset_y)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

    aligned_kpts = scaled_kpts + mean_offset

    return canvas, aligned_kpts


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to extract armor from clothed image."""
    result = image.copy()
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    result[:, :, 3] = np.minimum(result[:, :, 3], mask)
    return result


# ============ Step 2: Rigid Rotation ============

def _get_bone_angle(joint: np.ndarray, child: np.ndarray) -> float:
    """Get angle of bone from joint to child."""
    delta = child - joint
    return math.atan2(delta[1], delta[0])


def _create_segment_mask(
    h: int, w: int,
    joint_pos: np.ndarray,
    child_pos: np.ndarray,
    width: int
) -> np.ndarray:
    """Create mask for a limb segment."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(joint_pos[0]), int(joint_pos[1]))
    pt2 = (int(child_pos[0]), int(child_pos[1]))
    cv2.line(mask, pt1, pt2, 255, width)
    cv2.circle(mask, pt1, width // 2, 255, -1)
    cv2.circle(mask, pt2, width // 2, 255, -1)
    return mask > 0


def _rotate_point(point: np.ndarray, pivot: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate point around pivot."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    p = point - pivot
    rotated = np.array([p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a])
    return rotated + pivot


def apply_rotation(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    target_kpts: np.ndarray,
    config: TransformConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rigid rotation at joints to match target skeleton.

    Args:
        armor: Armor RGBA image
        armor_kpts: Current armor keypoints
        target_kpts: Target keypoints to match
        config: Transform configuration

    Returns:
        (rotated_armor, rotated_keypoints)
    """
    h, w = armor.shape[:2]
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    for chain in LIMB_CHAINS:
        for joint_idx, child_idx, name in chain:
            current_angle = _get_bone_angle(result_kpts[joint_idx], result_kpts[child_idx])
            target_angle = _get_bone_angle(target_kpts[joint_idx], target_kpts[child_idx])
            delta_angle = target_angle - current_angle

            if abs(delta_angle) < 0.01:
                continue

            segment_mask = _create_segment_mask(
                h, w, result_kpts[joint_idx], result_kpts[child_idx],
                config.rotation_segment_width
            )
            armor_in_region = (result[:, :, 3] > 128) & segment_mask

            if not np.any(armor_in_region):
                continue

            pivot = result_kpts[joint_idx]

            # Clear region from result
            for c in range(4):
                result[:, :, c] = np.where(armor_in_region, 0, result[:, :, c])

            # Extract segment
            segment_img = armor.copy()
            for c in range(4):
                segment_img[:, :, c] = np.where(armor_in_region, armor[:, :, c], 0)

            # Rotate segment
            angle_deg = math.degrees(delta_angle)
            M = cv2.getRotationMatrix2D((pivot[0], pivot[1]), -angle_deg, 1.0)
            rotated_segment = cv2.warpAffine(
                segment_img, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            # Composite rotated segment back
            rot_alpha = rotated_segment[:, :, 3:4] / 255.0
            for c in range(3):
                result[:, :, c] = (rotated_segment[:, :, c] * rot_alpha[:, :, 0] +
                                  result[:, :, c] * (1 - rot_alpha[:, :, 0])).astype(np.uint8)
            result[:, :, 3] = np.maximum(result[:, :, 3], rotated_segment[:, :, 3])

            # Update keypoint
            result_kpts[child_idx] = _rotate_point(result_kpts[child_idx], pivot, delta_angle)
            armor = result.copy()

    return result, result_kpts


# ============ Step 3: Soft-Edge Inpaint ============

def _get_uncovered_mask(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int
) -> np.ndarray:
    """Get mask of base pixels not covered by armor."""
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers

    # Only below neck
    h = base_image.shape[0]
    valid_region = np.zeros((h, base_image.shape[1]), dtype=bool)
    valid_region[neck_y:, :] = True

    return uncovered & valid_region


def _get_armor_edge_near_uncovered(
    armor_alpha: np.ndarray,
    uncovered_mask: np.ndarray,
    edge_width: int
) -> np.ndarray:
    """Find armor edge pixels adjacent to uncovered areas."""
    armor_mask = armor_alpha > 128
    dilated_uncovered = binary_dilation(uncovered_mask, iterations=edge_width)
    armor_edge_near_uncovered = armor_mask & dilated_uncovered
    eroded_armor = binary_erosion(armor_mask, iterations=1)
    armor_edge = armor_mask & ~eroded_armor
    return armor_edge_near_uncovered | (armor_edge & dilated_uncovered)


def apply_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    base_image: np.ndarray,
    armor_kpts: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: TransformConfig
) -> np.ndarray:
    """Apply soft-edge texture inpainting.

    Args:
        armor: Current armor RGBA
        original_clothed: Original scaled/aligned clothed image (for texture sampling)
        base_image: Base frame RGBA
        armor_kpts: Armor keypoints (after rotation)
        base_kpts: Base keypoints
        armor_mask: Original armor mask
        config: Transform configuration

    Returns:
        Inpainted armor image
    """
    neck_y = int(base_kpts[1, 1])
    uncovered = _get_uncovered_mask(base_image, armor, neck_y)

    if not np.any(uncovered):
        return armor

    armor_edge = _get_armor_edge_near_uncovered(armor[:, :, 3], uncovered, config.edge_width)

    result = armor.copy()
    result[:, :, 3] = np.where(armor_edge, 0, armor[:, :, 3])

    inpaint_region = uncovered | armor_edge

    if not np.any(inpaint_region):
        return armor

    # TPS mapping for texture borrowing
    h, w = armor.shape[:2]
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float64)
    src_all = np.vstack([armor_kpts, corners])
    dst_all = np.vstack([base_kpts, corners])

    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0)

    inpaint_ys, inpaint_xs = np.where(inpaint_region)
    dst_coords = np.column_stack([inpaint_xs, inpaint_ys])
    src_xs = rbf_x(dst_coords)
    src_ys = rbf_y(dst_coords)

    for i, (dst_y, dst_x) in enumerate(zip(inpaint_ys, inpaint_xs)):
        src_x = int(round(src_xs[i]))
        src_y = int(round(src_ys[i]))

        # Try TPS-mapped position
        if 0 <= src_x < w and 0 <= src_y < h:
            if armor_mask[src_y, src_x] > 128:
                result[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result[dst_y, dst_x, 3] = 255
                continue

        # Fallback: nearest armor pixel
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


# ============ Step 4: Pixelize ============

def apply_pixelize(image: np.ndarray, factor: int) -> np.ndarray:
    """Pixelize by downscale then upscale with nearest neighbor."""
    if factor <= 1:
        return image

    h, w = image.shape[:2]
    small_h, small_w = h // factor, w // factor

    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return result


# ============ Full Pipeline ============

@dataclass
class TransformDebugOutput:
    """Debug outputs from transform pipeline."""
    aligned_clothed: np.ndarray       # After scale + align
    aligned_kpts: np.ndarray          # Keypoints after scale + align
    armor_masked: np.ndarray          # After applying mask
    rotated_armor: np.ndarray         # After rotation
    rotated_kpts: np.ndarray          # Keypoints after rotation
    inpainted_armor: np.ndarray       # After inpaint
    final_armor: np.ndarray           # After pixelize
    overlap_viz: np.ndarray           # Blue/red/green overlap visualization
    skeleton_viz: np.ndarray          # Skeleton overlay


def _draw_skeleton_on_image(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4
) -> np.ndarray:
    """Draw skeleton on image."""
    from .keypoints import SKELETON_CONNECTIONS

    result = image.copy()
    if result.shape[2] == 4:
        # Convert RGBA to RGB for drawing
        rgb = result[:, :, :3].copy()
    else:
        rgb = result.copy()

    # Draw bones
    for i, j in SKELETON_CONNECTIONS:
        pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        cv2.line(rgb, pt1, pt2, color, thickness)

    # Draw points
    for i in range(len(keypoints)):
        pt = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        cv2.circle(rgb, pt, point_radius, color, -1)
        cv2.circle(rgb, pt, point_radius, (255, 255, 255), 1)

    return rgb


def _create_overlap_visualization(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int
) -> np.ndarray:
    """Create blue/red/green overlap visualization.

    - Green: armor covering base (good)
    - Blue: base not covered by armor (uncovered)
    - Red: armor not over base (floating)
    """
    h, w = base_image.shape[:2]
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    base_visible = base_image[:, :, 3] > 128
    armor_visible = armor_image[:, :, 3] > 128

    # Only consider below neck
    valid_region = np.zeros((h, w), dtype=bool)
    valid_region[neck_y:, :] = True

    # Green: armor covers base
    covered = base_visible & armor_visible & valid_region
    viz[covered] = [0, 255, 0]

    # Blue: base not covered by armor
    uncovered = base_visible & ~armor_visible & valid_region
    viz[uncovered] = [255, 0, 0]  # BGR so this is blue

    # Red: floating armor (armor but no base)
    floating = ~base_visible & armor_visible & valid_region
    viz[floating] = [0, 0, 255]  # BGR so this is red

    return viz


def _clean_ghost_pixels(image: np.ndarray, alpha_threshold: int = 128) -> np.ndarray:
    """Zero out RGB where alpha is below threshold to remove ghost pixels.

    Some source images have leftover RGB data in transparent areas from
    previous editing. This can cause ghost artifacts when scaling.
    """
    result = image.copy()
    ghost_mask = result[:, :, 3] < alpha_threshold
    result[ghost_mask, :3] = 0
    return result


def transform_frame(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None
) -> np.ndarray:
    """Run full transform pipeline on a single frame.

    Args:
        clothed_image: Clothed reference frame RGBA
        clothed_kpts: Keypoints for clothed frame
        base_image: Base frame RGBA
        base_kpts: Keypoints for base frame
        armor_mask: Mask separating armor from clothed image
        config: Transform configuration (uses defaults if None)

    Returns:
        Transformed armor RGBA image
    """
    if config is None:
        config = TransformConfig()

    # Pre-process: clean ghost pixels from transparent areas
    clothed_image = _clean_ghost_pixels(clothed_image)

    # Step 1: Scale and align
    aligned_clothed, aligned_kpts = scale_and_align(
        clothed_image, clothed_kpts, base_kpts, config
    )

    # Scale mask the same way
    mask_rgba = np.zeros((*armor_mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, 0] = armor_mask
    mask_rgba[:, :, 3] = armor_mask
    aligned_mask, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config)
    scaled_mask = aligned_mask[:, :, 0]

    # Extract armor
    armor = apply_mask(aligned_clothed, scaled_mask)

    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor, aligned_kpts, base_kpts, config)

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        rotated_armor, aligned_clothed, base_image,
        rotated_kpts, base_kpts, scaled_mask, config
    )

    # Step 4: Pixelize
    final_armor = apply_pixelize(inpainted_armor, config.pixelize_factor)

    return final_armor


def transform_frame_debug(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None
) -> TransformDebugOutput:
    """Run full transform pipeline with debug outputs.

    Same as transform_frame but returns all intermediate steps.
    """
    if config is None:
        config = TransformConfig()

    # Pre-process: clean ghost pixels from transparent areas
    clothed_image = _clean_ghost_pixels(clothed_image)

    # Step 1: Scale and align
    aligned_clothed, aligned_kpts = scale_and_align(
        clothed_image, clothed_kpts, base_kpts, config
    )

    # Scale mask the same way
    mask_rgba = np.zeros((*armor_mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, 0] = armor_mask
    mask_rgba[:, :, 3] = armor_mask
    aligned_mask, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config)
    scaled_mask = aligned_mask[:, :, 0]

    # Extract armor
    armor_masked = apply_mask(aligned_clothed, scaled_mask)

    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor_masked, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor_masked, aligned_kpts, base_kpts, config)

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        rotated_armor, aligned_clothed, base_image,
        rotated_kpts, base_kpts, scaled_mask, config
    )

    # Step 4: Pixelize
    final_armor = apply_pixelize(inpainted_armor, config.pixelize_factor)

    # Create visualizations
    neck_y = int(base_kpts[1, 1])
    overlap_viz = _create_overlap_visualization(base_image, final_armor, neck_y)

    # Skeleton visualization: base skeleton (green) + armor skeleton (red) on base image
    skeleton_viz = base_image[:, :, :3].copy()
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, base_kpts, color=(0, 255, 0), thickness=2)
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, rotated_kpts, color=(0, 0, 255), thickness=1)

    return TransformDebugOutput(
        aligned_clothed=aligned_clothed,
        aligned_kpts=aligned_kpts,
        armor_masked=armor_masked,
        rotated_armor=rotated_armor,
        rotated_kpts=rotated_kpts,
        inpainted_armor=inpainted_armor,
        final_armor=final_armor,
        overlap_viz=overlap_viz,
        skeleton_viz=skeleton_viz
    )
