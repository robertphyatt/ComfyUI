"""Transform pipeline: scale, align, rotate, inpaint, pixelize."""

import cv2
import numpy as np
import math
import os
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import binary_dilation, binary_erosion

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


def compute_torso_center(keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
    """Compute center point from shoulders and hips.

    Uses keypoints 2 (left_shoulder), 3 (right_shoulder),
    10 (left_hip), 11 (right_hip) to define a stable torso center
    that's independent of limb positions or armor shape.

    Args:
        keypoints: (N, 2) array of keypoint coordinates

    Returns:
        (x, y) center or None if keypoints invalid
    """
    # Keypoint indices
    L_SHOULDER, R_SHOULDER = 2, 3
    L_HIP, R_HIP = 10, 11

    # Check we have valid keypoints (non-zero)
    points = keypoints[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
    if np.any(points == 0):
        return None

    # Average of all 4 points
    center = np.mean(points, axis=0)
    return (float(center[0]), float(center[1]))


def apply_torso_constraint(
    armor: np.ndarray,
    base_kpts: np.ndarray,
    armor_kpts: np.ndarray,
    anchor_base_center: Optional[Tuple[float, float]],
    anchor_armor_center: Optional[Tuple[float, float]]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Shift armor so its torso center delta matches base body's delta.

    Uses shoulder/hip keypoints to define torso center, which is stable
    across different armor shapes and limb positions.

    Args:
        armor: Current armor image (RGBA)
        base_kpts: Base body keypoints (N, 2)
        armor_kpts: Armor keypoints after transform (N, 2)
        anchor_base_center: Base torso center from frame 0 (None to skip)
        anchor_armor_center: Armor torso center from frame 0 (None to skip)

    Returns:
        Tuple of (shifted_armor, base_center, armor_center)
        - armor_center is the FINAL center after any shift applied
    """
    # Compute current torso centers from keypoints
    current_base_center = compute_torso_center(base_kpts)
    current_armor_center = compute_torso_center(armor_kpts)

    # If we can't compute centers, return original with None centers
    if current_base_center is None or current_armor_center is None:
        return armor, current_base_center, current_armor_center

    # Skip constraint if no anchor data (frame 0) - just return current centers
    if anchor_base_center is None or anchor_armor_center is None:
        return armor, current_base_center, current_armor_center

    # How much did base body move from frame 0?
    base_delta_x = current_base_center[0] - anchor_base_center[0]
    base_delta_y = current_base_center[1] - anchor_base_center[1]

    # Where should armor center be?
    expected_armor_x = anchor_armor_center[0] + base_delta_x
    expected_armor_y = anchor_armor_center[1] + base_delta_y

    # How much do we need to shift armor?
    shift_x = int(round(expected_armor_x - current_armor_center[0]))
    shift_y = int(round(expected_armor_y - current_armor_center[1]))

    # Skip if no shift needed
    if shift_x == 0 and shift_y == 0:
        return armor, current_base_center, current_armor_center

    # Apply shift using translation matrix
    h, w = armor.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(armor, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # The new armor center is the expected position (what we shifted to)
    final_armor_center = (expected_armor_x, expected_armor_y)

    return shifted, current_base_center, final_armor_center


@dataclass
class TransformConfig:
    """Configuration for the transform pipeline."""
    scale_factor: float = 1.0  # Was 1.057 but caused 541px overflow on 512px canvas
    rotation_segment_width: int = 35
    refinement_segment_width: int = 50  # Wider than rotation to capture shoulder/joint areas
    edge_width: int = 2
    pixelize_factor: int = 2  # Less blocky for higher-quality pixel art
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
    config: TransformConfig,
    anchor_offset: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Scale image and align using mean of neck and hip offsets.

    Args:
        image: Source RGBA image
        image_kpts: Keypoints for source image
        target_kpts: Target keypoints to align to
        config: Transform configuration
        anchor_offset: If provided, use this offset instead of computing from keypoints.
                       This ensures consistent positioning across all frames in an animation.

    Returns:
        (aligned_image, aligned_keypoints, offset_used) - offset_used is (offset_x, offset_y)
    """
    h, w = image.shape[:2]
    scale = config.scale_factor
    canvas_size = config.canvas_size

    # Scale
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_kpts = image_kpts * scale

    if anchor_offset is not None:
        # Use provided anchor offset for consistent positioning across frames
        offset_x, offset_y = anchor_offset
        mean_offset = np.array([offset_x, offset_y], dtype=np.float64)
    else:
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

    return canvas, aligned_kpts, (offset_x, offset_y)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to extract armor from clothed image."""
    result = image.copy()
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    result[:, :, 3] = np.minimum(result[:, :, 3], mask)
    return result


def extend_mask_to_edges(
    mask: np.ndarray,
    clothed_alpha: np.ndarray,
    max_gap: int = 3
) -> np.ndarray:
    """Extend mask to fill thin gaps between mask edge and transparent pixels.

    When the armor mask is slightly smaller than the clothed sprite silhouette,
    thin strips of edge pixels get excluded. This extends the mask to include
    those pixels, but ONLY if they're:
    1. Adjacent to existing mask coverage (not floating elsewhere)
    2. Near external transparency (the sprite's outer edge)

    Args:
        mask: Original armor mask (0 or 255)
        clothed_alpha: Alpha channel of clothed image
        max_gap: Maximum gap width to fill (pixels)

    Returns:
        Extended mask
    """
    from scipy.ndimage import binary_dilation

    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask_visible = mask > 128
    clothed_visible = clothed_alpha > 128

    # Gap candidates: pixels in clothed sprite but not in mask
    gap_candidates = clothed_visible & ~mask_visible

    # Only extend to pixels that are ADJACENT to existing mask
    # This prevents extending to disconnected regions (like head outline)
    near_mask = binary_dilation(mask_visible, iterations=max_gap)
    adjacent_to_mask = gap_candidates & near_mask

    # Also require being near transparency (the outer edge of the sprite)
    transparent = clothed_alpha < 128
    near_transparency = binary_dilation(transparent, iterations=max_gap)

    # Thin strip = must be adjacent to mask AND near transparency
    thin_strip = adjacent_to_mask & near_transparency

    # Extended mask = original + thin strips
    extended = mask_visible | thin_strip

    return (extended * 255).astype(np.uint8)


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


# ============ Step 2.5: Silhouette Refinement ============

def _get_armor_segment_mask(
    armor: np.ndarray,
    keypoints: np.ndarray,
    joint_idx: int,
    child_idx: int,
    segment_width: int = 35
) -> np.ndarray:
    """Get mask of armor pixels belonging to a limb segment.

    Args:
        armor: Armor RGBA image
        keypoints: Current keypoints array
        joint_idx: Index of parent joint
        child_idx: Index of child joint
        segment_width: Width of segment region

    Returns:
        Boolean mask where armor pixels are within segment region
    """
    h, w = armor.shape[:2]

    # Get segment region (same as rotation uses)
    segment_region = _create_segment_mask(
        h, w, keypoints[joint_idx], keypoints[child_idx], segment_width
    )

    # Intersect with actual armor pixels
    armor_visible = armor[:, :, 3] > 128

    return armor_visible & segment_region


def _apply_translation(
    armor: np.ndarray,
    combined_mask: np.ndarray,
    offset: Tuple[int, int]
) -> np.ndarray:
    """Apply translation to masked pixels.

    Args:
        armor: Armor RGBA image
        combined_mask: Boolean mask of pixels to move
        offset: (dx, dy) translation offset

    Returns:
        Translated armor image
    """
    dx, dy = offset
    h, w = armor.shape[:2]
    result = armor.copy()

    # Extract pixels to move
    pixels_to_move = np.zeros_like(armor)
    for c in range(4):
        pixels_to_move[:, :, c] = np.where(combined_mask, armor[:, :, c], 0)

    # Clear original positions
    for c in range(4):
        result[:, :, c] = np.where(combined_mask, 0, result[:, :, c])

    # Translate using warpAffine
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(
        pixels_to_move, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Composite translated pixels back
    trans_alpha = translated[:, :, 3:4] / 255.0
    for c in range(3):
        result[:, :, c] = (translated[:, :, c] * trans_alpha[:, :, 0] +
                          result[:, :, c] * (1 - trans_alpha[:, :, 0])).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], translated[:, :, 3])

    return result


def _translate_segment(
    armor: np.ndarray,
    keypoints: np.ndarray,
    segment_mask: np.ndarray,
    offset: Tuple[int, int],
    descendant_masks: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate armor pixels in segment and all descendants by offset.

    Note: This function only translates pixels, not keypoints. The caller
    is responsible for updating keypoints after translation.

    Args:
        armor: Armor RGBA image
        keypoints: Current keypoints array (returned unchanged)
        segment_mask: Boolean mask of segment pixels to move
        offset: (dx, dy) translation offset
        descendant_masks: List of masks for descendant segments (also moved)

    Returns:
        (translated_armor, keypoints_copy)
    """
    dx, dy = offset
    if dx == 0 and dy == 0:
        return armor, keypoints

    # Combine segment mask with all descendant masks
    combined_mask = segment_mask.copy()
    for desc_mask in descendant_masks:
        combined_mask = combined_mask | desc_mask

    result = _apply_translation(armor, combined_mask, offset)
    return result, keypoints.copy()


def _try_1px_move(
    armor: np.ndarray,
    base_image: np.ndarray,
    segment_mask: np.ndarray,
    descendant_masks: List[np.ndarray],
    prev_red: int,
    prev_blue: int
) -> Tuple[Tuple[int, int], int, int]:
    """Try 1px moves in 8 directions, return first that improves global metrics.

    Args:
        armor: Current armor RGBA image
        base_image: Base frame RGBA image
        segment_mask: Mask of pixels in this segment
        descendant_masks: Masks of descendant segments (moved together)
        prev_red: Current global red pixel count
        prev_blue: Current global blue pixel count

    Returns:
        ((dx, dy), new_red, new_blue) - the move and resulting metrics
        ((0, 0), prev_red, prev_blue) if no valid move found
    """
    base_visible = base_image[:, :, 3] > 128

    # Combine segment mask with descendants
    combined_mask = segment_mask.copy()
    for desc_mask in descendant_masks:
        combined_mask = combined_mask | desc_mask

    # 8 directions: cardinal + diagonal
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        # Try this translation
        test_armor = _apply_translation(armor, combined_mask, (dx, dy))

        # Check global metrics
        armor_visible = test_armor[:, :, 3] > 128
        new_red = int(np.sum(armor_visible & ~base_visible))
        new_blue = int(np.sum(base_visible & ~armor_visible))

        # Accept if neither increased AND at least one improved
        if new_red <= prev_red and new_blue <= prev_blue and (new_red < prev_red or new_blue < prev_blue):
            return (dx, dy), new_red, new_blue

    # No valid move found
    return (0, 0), prev_red, prev_blue


def refine_silhouette_alignment(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    config: TransformConfig,
    max_iterations: int = 50  # Unused, kept for API compatibility
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively refine armor position using greedy 1px per-segment moves.

    Each segment attempts one 1px move per round. Moves are accepted only if
    global red AND blue don't increase (and at least one improves). Parent
    segments cascade movement to children. Loop continues until no segment
    can make a valid move.

    Args:
        armor: Rotated armor RGBA image
        armor_kpts: Armor keypoints after rotation
        base_image: Base frame RGBA image
        base_kpts: Base frame keypoints (unused, kept for API compatibility)
        config: Transform configuration
        max_iterations: Unused, kept for API compatibility

    Returns:
        (refined_armor, refined_keypoints)
    """
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    # Compute initial global metrics
    base_visible = base_image[:, :, 3] > 128
    armor_visible = result[:, :, 3] > 128
    curr_red = int(np.sum(armor_visible & ~base_visible))
    curr_blue = int(np.sum(base_visible & ~armor_visible))

    # Loop until no segment can improve
    while True:
        any_moved = False

        # Process all limb chains hierarchically
        for chain in LIMB_CHAINS:
            for i, (joint_idx, child_idx, name) in enumerate(chain):
                # Get segment mask (use wider refinement width to capture shoulder areas)
                segment_mask = _get_armor_segment_mask(
                    result, result_kpts, joint_idx, child_idx,
                    config.refinement_segment_width
                )

                if not np.any(segment_mask):
                    continue

                # Get descendant masks (remaining segments in chain)
                descendant_masks = []
                for j in range(i + 1, len(chain)):
                    desc_joint, desc_child, _ = chain[j]
                    desc_mask = _get_armor_segment_mask(
                        result, result_kpts, desc_joint, desc_child,
                        config.refinement_segment_width
                    )
                    if np.any(desc_mask):
                        descendant_masks.append(desc_mask)

                # Try 1px move in each direction
                offset, new_red, new_blue = _try_1px_move(
                    result, base_image, segment_mask, descendant_masks,
                    curr_red, curr_blue
                )

                if offset != (0, 0):
                    # Apply the move
                    result, result_kpts = _translate_segment(
                        result, result_kpts, segment_mask, offset, descendant_masks
                    )

                    # Update keypoints for this segment and descendants
                    dx, dy = offset
                    result_kpts[child_idx] = result_kpts[child_idx] + np.array([dx, dy])
                    for j in range(i + 1, len(chain)):
                        desc_joint, desc_child, _ = chain[j]
                        result_kpts[desc_child] = result_kpts[desc_child] + np.array([dx, dy])

                    # Update metrics
                    curr_red = new_red
                    curr_blue = new_blue
                    any_moved = True

        # If no segment moved this round, we're converged
        if not any_moved:
            break

    return result, result_kpts


# ============ Step 3: Soft-Edge Inpaint ============

def _get_uncovered_mask(
    base_image: np.ndarray,
    armor_image: np.ndarray,
    neck_y: int,
    clothing_mask: np.ndarray = None
) -> np.ndarray:
    """Get mask of base pixels not covered by armor.

    Args:
        base_image: Base frame RGBA
        armor_image: Armor frame RGBA
        neck_y: Y coordinate of neck keypoint (fallback if no mask)
        clothing_mask: Optional mask defining valid inpaint region (preferred)
    """
    base_visible = base_image[:, :, 3] > 128
    armor_covers = armor_image[:, :, 3] > 128
    uncovered = base_visible & ~armor_covers

    h, w = base_image.shape[:2]

    if clothing_mask is not None:
        # Use mask to define valid region - only inpaint within masked area
        # Dilate mask slightly to include edges
        valid_region = binary_dilation(clothing_mask > 128, iterations=3)
    else:
        # Fallback: only below neck (works for back view)
        valid_region = np.zeros((h, w), dtype=bool)
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


def get_interior_mask(alpha: np.ndarray, erosion: int = 2) -> np.ndarray:
    """Get mask of interior pixels safe to sample from (not near edges).

    Args:
        alpha: Alpha channel (0-255)
        erosion: How many pixels to erode from edges

    Returns:
        Boolean mask where True = interior pixel
    """
    from scipy.ndimage import binary_erosion

    visible = alpha > 128
    interior = binary_erosion(visible, iterations=erosion)

    return interior


def detect_outline_pixels(
    image: np.ndarray,
    brightness_threshold: int = 60,
    contrast_threshold: int = 50
) -> np.ndarray:
    """Detect outline pixels that should be inpainted over.

    An outline pixel is one that:
    1. Is relatively dark (below brightness_threshold), AND
    2. Is adjacent to transparency OR adjacent to a much brighter pixel

    This preserves dark interior details (shadows, leather) while
    catching outline artifacts from rotation.

    Args:
        image: BGRA image (quantized to palette)
        brightness_threshold: Max brightness to consider as potential outline (0-255)
        contrast_threshold: Min brightness difference to neighbor to trigger

    Returns:
        Boolean mask where True = outline pixel to inpaint
    """
    h, w = image.shape[:2]
    alpha = image[:, :, 3]
    visible = alpha > 128

    # Calculate brightness (simple average of BGR)
    brightness = np.mean(image[:, :, :3], axis=2).astype(np.float32)

    # Dark pixels are candidates
    is_dark = brightness < brightness_threshold

    # Check adjacency to transparency
    transparent = alpha <= 128
    adjacent_to_transparent = binary_dilation(transparent, iterations=1) & visible

    # Check adjacency to much brighter pixels
    # Dilate brightness and check if any neighbor is much brighter
    from scipy.ndimage import maximum_filter
    max_neighbor_brightness = maximum_filter(brightness, size=3)
    adjacent_to_bright = (max_neighbor_brightness - brightness) > contrast_threshold

    # Outline = dark AND (adjacent to transparency OR adjacent to bright)
    outline_mask = is_dark & visible & (adjacent_to_transparent | adjacent_to_bright)

    return outline_mask


def apply_inpaint(
    armor: np.ndarray,
    original_clothed: np.ndarray,
    base_image: np.ndarray,
    armor_kpts: np.ndarray,
    base_kpts: np.ndarray,
    config: TransformConfig,
    clothing_mask: np.ndarray = None
) -> np.ndarray:
    """Apply soft-edge texture inpainting.

    Args:
        armor: Current armor RGBA
        original_clothed: Original scaled/aligned clothed image (for texture sampling)
        base_image: Base frame RGBA
        armor_kpts: Armor keypoints (after rotation)
        base_kpts: Base keypoints
        config: Transform configuration
        clothing_mask: Optional mask defining valid inpaint region (preferred over neck_y)

    Returns:
        Inpainted armor image
    """
    neck_y = int(base_kpts[1, 1])
    uncovered = _get_uncovered_mask(base_image, armor, neck_y, clothing_mask)

    # Create interior mask for safe sampling (avoid edge pixels)
    # Use armor's actual alpha, not the mask - we want to avoid armor's edge pixels
    interior_mask = get_interior_mask(armor[:, :, 3], erosion=2)

    if not np.any(uncovered):
        return armor

    armor_edge = _get_armor_edge_near_uncovered(armor[:, :, 3], uncovered, config.edge_width)

    # Detect outline pixels that should be inpainted over
    # These are dark pixels adjacent to transparency or bright pixels
    outline_pixels = detect_outline_pixels(armor)

    result = armor.copy()
    # Clear both armor edge and outline pixels - they'll be inpainted
    clear_mask = armor_edge | outline_pixels
    result[:, :, 3] = np.where(clear_mask, 0, armor[:, :, 3])

    inpaint_region = uncovered | armor_edge | outline_pixels

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
            if armor[src_y, src_x, 3] > 128 and interior_mask[src_y, src_x]:
                result[dst_y, dst_x, :3] = original_clothed[src_y, src_x, :3]
                result[dst_y, dst_x, 3] = 255
                continue

        # Fallback: nearest interior armor pixel (prefer interior, fall back to any)
        orig_armor_alpha = armor[:, :, 3] > 128
        for radius in range(1, 30):
            y1, y2 = max(0, dst_y - radius), min(h, dst_y + radius + 1)
            x1, x2 = max(0, dst_x - radius), min(w, dst_x + radius + 1)

            # Prefer interior pixels
            box_interior = orig_armor_alpha[y1:y2, x1:x2] & interior_mask[y1:y2, x1:x2]
            if np.any(box_interior):
                box_ys, box_xs = np.where(box_interior)
                abs_ys, abs_xs = box_ys + y1, box_xs + x1
                distances = (abs_ys - dst_y) ** 2 + (abs_xs - dst_x) ** 2
                closest = np.argmin(distances)
                result[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                result[dst_y, dst_x, 3] = 255
                break

            # Fall back to any armor pixel if no interior found
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
    refined_armor: np.ndarray         # After silhouette refinement
    refined_kpts: np.ndarray          # Keypoints after refinement
    inpainted_armor: np.ndarray       # After inpaint
    final_armor: np.ndarray           # After inpaint (pre-pixelize)
    pre_inpaint_overlap_viz: np.ndarray  # Overlap viz BEFORE inpaint (shows what needs filling)
    post_refine_overlap_viz: np.ndarray  # Overlap viz AFTER refinement (shows improvement)
    overlap_viz: np.ndarray           # Blue/red/green overlap visualization (after inpaint)
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
    neck_y: int,
    clothing_mask: np.ndarray = None
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

    # Use same logic as _get_uncovered_mask for consistency
    if clothing_mask is not None:
        valid_region = binary_dilation(clothing_mask > 128, iterations=3)
    else:
        # Fallback: only below neck (works for back view)
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

    # DEBUG LOGGING - triggered by environment variable
    if os.environ.get("DEBUG_OVERLAP") == "1":
        debug_dir = "/tmp/overlap_debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/base_image.png", base_image)
        cv2.imwrite(f"{debug_dir}/armor_image.png", armor_image)
        cv2.imwrite(f"{debug_dir}/base_visible.png", (base_visible * 255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/armor_visible.png", (armor_visible * 255).astype(np.uint8))
        print(f"[DEBUG overlap] neck_y={neck_y}")
        print(f"[DEBUG overlap] base_visible pixels: {np.sum(base_visible)}")
        print(f"[DEBUG overlap] armor_visible pixels: {np.sum(armor_visible)}")
        print(f"[DEBUG overlap] covered (green): {np.sum(covered)}")
        print(f"[DEBUG overlap] uncovered (blue): {np.sum(uncovered)}")
        print(f"[DEBUG overlap] floating (red): {np.sum(floating)}")
        print(f"[DEBUG overlap] Saved debug images to {debug_dir}/")

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
    config: Optional[TransformConfig] = None,
    anchor_offset: Optional[Tuple[int, int]] = None,
    anchor_base_center: Optional[Tuple[float, float]] = None,
    anchor_armor_center: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Tuple[int, int], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Run full transform pipeline on a single frame.

    Args:
        clothed_image: Clothed reference frame RGBA
        clothed_kpts: Keypoints for clothed frame
        base_image: Base frame RGBA
        base_kpts: Keypoints for base frame
        armor_mask: Mask separating armor from clothed image
        config: Transform configuration (uses defaults if None)
        anchor_offset: If provided, use this offset for alignment instead of computing.
                       Pass the offset from frame 0 to ensure consistent positioning.
        anchor_base_center: Base torso center from frame 0 (None for frame 0)
        anchor_armor_center: Armor torso center from frame 0 (None for frame 0)

    Returns:
        (final_armor, offset_used, base_center, armor_center)
        - Centers are from shoulder/hip keypoints, returned for frame 0 to establish anchors
    """
    if config is None:
        config = TransformConfig()

    # Pre-process: clean ghost pixels from transparent areas
    clothed_image = _clean_ghost_pixels(clothed_image)

    # Step 1: Scale and align
    aligned_clothed, aligned_kpts, offset_used = scale_and_align(
        clothed_image, clothed_kpts, base_kpts, config, anchor_offset
    )

    # Scale mask the same way (using the same offset for consistency)
    mask_rgba = np.zeros((*armor_mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, 0] = armor_mask
    mask_rgba[:, :, 3] = armor_mask
    aligned_mask, _, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config, offset_used)
    scaled_mask = aligned_mask[:, :, 0]

    # Extend mask to include thin edge strips near transparency
    extended_mask = extend_mask_to_edges(scaled_mask, aligned_clothed[:, :, 3])

    # Extract armor
    armor = apply_mask(aligned_clothed, extended_mask)

    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor, aligned_kpts, base_kpts, config)

    # Step 2.25: Torso constraint (BEFORE refinement)
    # Align torso center before refinement works on limbs
    # Capture the centers here - these are the correct pre-refinement values
    rotated_armor, current_base_center, current_armor_center = apply_torso_constraint(
        rotated_armor, base_kpts, rotated_kpts,
        anchor_base_center, anchor_armor_center
    )

    # Step 2.5: Silhouette refinement
    # Refinement needs access to FULL armor (including parts extending outside base)
    # so it can shift segments to cover gaps in the base silhouette.
    refined_armor, refined_kpts = refine_silhouette_alignment(
        rotated_armor, rotated_kpts, base_image, base_kpts, config
    )

    # Step 2.6: Apply base silhouette mask AFTER refinement
    # Now crop to base silhouette - refinement has already positioned segments optimally
    base_silhouette = base_image[:, :, 3] > 128
    refined_armor = apply_mask(refined_armor, (base_silhouette * 255).astype(np.uint8))

    # Step 3: Inpaint
    final_armor = apply_inpaint(
        refined_armor, aligned_clothed, base_image,
        refined_kpts, base_kpts, config,
        clothing_mask=armor_mask
    )

    # Pixelization now happens after color correction in pipeline
    # Return centers captured at step 2.25 (pre-refinement, which is correct)
    return final_armor, offset_used, current_base_center, current_armor_center


def transform_frame_debug(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None,
    anchor_offset: Optional[Tuple[int, int]] = None,
    anchor_base_center: Optional[Tuple[float, float]] = None,
    anchor_armor_center: Optional[Tuple[float, float]] = None
) -> Tuple[TransformDebugOutput, Tuple[int, int], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Run full transform pipeline with debug outputs.

    Same as transform_frame but returns all intermediate steps.

    Returns:
        (debug_output, offset_used, base_center, armor_center)
    """
    if config is None:
        config = TransformConfig()

    # Pre-process: clean ghost pixels from transparent areas
    clothed_image = _clean_ghost_pixels(clothed_image)

    # Step 1: Scale and align
    aligned_clothed, aligned_kpts, offset_used = scale_and_align(
        clothed_image, clothed_kpts, base_kpts, config, anchor_offset
    )

    # Scale mask the same way (using the same offset for consistency)
    mask_rgba = np.zeros((*armor_mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, 0] = armor_mask
    mask_rgba[:, :, 3] = armor_mask
    aligned_mask, _, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config, offset_used)
    scaled_mask = aligned_mask[:, :, 0]

    # Extend mask to include thin edge strips near transparency
    extended_mask = extend_mask_to_edges(scaled_mask, aligned_clothed[:, :, 3])

    # Extract armor
    armor_masked = apply_mask(aligned_clothed, extended_mask)

    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor_masked, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor_masked, aligned_kpts, base_kpts, config)

    # Step 2.25: Torso constraint (BEFORE refinement)
    # Align torso center before refinement works on limbs
    # Capture the centers here - these are the correct pre-refinement values
    rotated_armor, current_base_center, current_armor_center = apply_torso_constraint(
        rotated_armor, base_kpts, rotated_kpts,
        anchor_base_center, anchor_armor_center
    )

    # Step 2.5: Silhouette refinement
    # Refinement needs access to FULL armor (including parts extending outside base)
    # so it can shift segments to cover gaps in the base silhouette.
    refined_armor, refined_kpts = refine_silhouette_alignment(
        rotated_armor, rotated_kpts, base_image, base_kpts, config
    )

    # Step 2.6: Apply base silhouette mask AFTER refinement
    # Now crop to base silhouette - refinement has already positioned segments optimally
    base_silhouette = base_image[:, :, 3] > 128
    refined_armor = apply_mask(refined_armor, (base_silhouette * 255).astype(np.uint8))

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        refined_armor, aligned_clothed, base_image,
        refined_kpts, base_kpts, config,
        clothing_mask=armor_mask
    )

    final_armor = inpainted_armor

    # Centers were captured at step 2.25 (pre-refinement, which is correct)

    # Create visualizations
    neck_y = int(base_kpts[1, 1])

    # Pre-inpaint overlap shows what needs to be filled (before refinement)
    pre_inpaint_overlap_viz = _create_overlap_visualization(base_image, rotated_armor, neck_y, clothing_mask=armor_mask)

    # Post-refinement overlap shows improvement from silhouette alignment
    post_refine_overlap_viz = _create_overlap_visualization(base_image, refined_armor, neck_y, clothing_mask=armor_mask)

    # Post-inpaint overlap shows final coverage (after inpainting)
    overlap_viz = _create_overlap_visualization(base_image, final_armor, neck_y, clothing_mask=armor_mask)

    # Skeleton visualization: base skeleton (green) + armor skeleton (red) on base image
    skeleton_viz = base_image[:, :, :3].copy()
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, base_kpts, color=(0, 255, 0), thickness=2)
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, refined_kpts, color=(0, 0, 255), thickness=1)

    return TransformDebugOutput(
        aligned_clothed=aligned_clothed,
        aligned_kpts=aligned_kpts,
        armor_masked=armor_masked,
        rotated_armor=rotated_armor,
        rotated_kpts=rotated_kpts,
        refined_armor=refined_armor,
        refined_kpts=refined_kpts,
        inpainted_armor=inpainted_armor,
        final_armor=final_armor,
        pre_inpaint_overlap_viz=pre_inpaint_overlap_viz,
        post_refine_overlap_viz=post_refine_overlap_viz,
        overlap_viz=overlap_viz,
        skeleton_viz=skeleton_viz
    ), offset_used, current_base_center, current_armor_center
