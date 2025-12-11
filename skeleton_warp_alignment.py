#!/usr/bin/env python3
"""Skeleton-guided warping using per-limb affine transformations."""

import numpy as np
import cv2
from typing import Dict, Tuple, List
from pathlib import Path
from PIL import Image
from align_with_openpose import extract_openpose_keypoints


# OpenPose keypoint indices
KEYPOINT_NAMES = {
    0: 'nose', 1: 'neck', 2: 'r_shoulder', 3: 'r_elbow', 4: 'r_wrist',
    5: 'l_shoulder', 6: 'l_elbow', 7: 'l_wrist', 8: 'r_hip', 9: 'r_knee',
    10: 'r_ankle', 11: 'l_hip', 12: 'l_knee', 13: 'l_ankle', 14: 'r_eye',
    15: 'l_eye', 16: 'r_ear', 17: 'l_ear'
}

# Define body regions as keypoint groups
BODY_REGIONS = {
    'head': [0, 1, 14, 15, 16, 17],  # nose, neck, eyes, ears
    'torso': [1, 2, 5, 8, 11],  # neck, shoulders, hips
    'right_upper_arm': [2, 3],  # r_shoulder, r_elbow
    'right_lower_arm': [3, 4],  # r_elbow, r_wrist
    'left_upper_arm': [5, 6],  # l_shoulder, l_elbow
    'left_lower_arm': [6, 7],  # l_elbow, l_wrist
    'right_upper_leg': [8, 9],  # r_hip, r_knee
    'right_lower_leg': [9, 10],  # r_knee, r_ankle
    'left_upper_leg': [11, 12],  # l_hip, l_knee
    'left_lower_leg': [12, 13],  # l_knee, l_ankle
}


def get_keypoint_positions(keypoints: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract keypoint positions and confidence values.

    Args:
        keypoints: OpenPose keypoints dict

    Returns:
        Tuple of (positions, confidences)
        - positions: Array of shape (18, 2) with x, y positions (normalized 0-1)
        - confidences: Array of shape (18,) with confidence values (0-1)
    """
    if not keypoints or 'people' not in keypoints or len(keypoints['people']) == 0:
        raise ValueError("No people detected in keypoints")

    person = keypoints['people'][0]
    kp = person['pose_keypoints_2d']

    # Extract (x, y, confidence) for each of 18 keypoints
    positions = np.zeros((18, 2), dtype=np.float32)
    confidences = np.zeros(18, dtype=np.float32)

    for i in range(18):
        positions[i, 0] = kp[i*3]      # x
        positions[i, 1] = kp[i*3 + 1]  # y
        confidences[i] = kp[i*3 + 2]   # confidence

    return positions, confidences


def get_region_transform(src_kp: np.ndarray, dst_kp: np.ndarray,
                         src_conf: np.ndarray, dst_conf: np.ndarray,
                         region_indices: List[int],
                         image_size: int = 512) -> Tuple[np.ndarray, bool]:
    """Calculate affine transform for a body region.

    Args:
        src_kp: Source keypoints (18, 2) normalized
        dst_kp: Destination keypoints (18, 2) normalized
        src_conf: Source confidences (18,)
        dst_conf: Destination confidences (18,)
        region_indices: List of keypoint indices for this region
        image_size: Image size in pixels

    Returns:
        (transform_matrix, valid) tuple
        - transform_matrix: 2x3 affine transform or identity if invalid
        - valid: Whether transform could be computed
    """
    # Filter to only keypoints in this region with good confidence
    valid_mask = np.array([
        (src_conf[i] > 0.1 and dst_conf[i] > 0.1)
        for i in region_indices
    ])

    if np.sum(valid_mask) < 2:
        # Need at least 2 points for affine transform
        return np.eye(2, 3, dtype=np.float32), False

    # Get valid keypoints for this region
    valid_indices = [region_indices[i] for i, v in enumerate(valid_mask) if v]
    src_pts = src_kp[valid_indices] * image_size
    dst_pts = dst_kp[valid_indices] * image_size

    if len(src_pts) == 2:
        # With 2 points, add a third point perpendicular to the line
        # This preserves the rotation and scale but allows for translation
        vec = src_pts[1] - src_pts[0]
        perp = np.array([-vec[1], vec[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6) * 10  # 10px perpendicular
        src_pts = np.vstack([src_pts, src_pts[0] + perp])

        vec = dst_pts[1] - dst_pts[0]
        perp = np.array([-vec[1], vec[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6) * 10
        dst_pts = np.vstack([dst_pts, dst_pts[0] + perp])

    # Compute affine transform
    try:
        M = cv2.getAffineTransform(src_pts[:3].astype(np.float32),
                                   dst_pts[:3].astype(np.float32))
        return M, True
    except cv2.error as e:
        # Affine transform failed (likely collinear points)
        return np.eye(2, 3, dtype=np.float32), False


def create_region_weights(image_size: int, keypoints: np.ndarray,
                         confidences: np.ndarray, region_indices: List[int],
                         sigma: float = 50.0) -> np.ndarray:
    """Create weight map for a body region based on distance to keypoints.

    Args:
        image_size: Image dimensions
        keypoints: Keypoint positions in pixels (18, 2)
        confidences: Keypoint confidence values (18,)
        region_indices: Keypoint indices for this region
        sigma: Gaussian falloff distance in pixels

    Returns:
        Weight map (image_size, image_size) with values 0-1
    """
    weights = np.zeros((image_size, image_size), dtype=np.float32)

    # Create coordinate grid
    y, x = np.mgrid[0:image_size, 0:image_size]
    coords = np.stack([x, y], axis=-1)  # (H, W, 2)

    # For each keypoint in region, add Gaussian weight
    for idx in region_indices:
        # Skip keypoints with low confidence
        if confidences[idx] < 0.1:
            continue

        kp = keypoints[idx]

        # Distance from each pixel to this keypoint
        dist = np.linalg.norm(coords - kp, axis=-1)

        # Gaussian weight
        weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        weights += weight

    # Normalize
    weights = np.clip(weights, 0, 1)
    return weights


def warp_with_per_limb_transforms(clothed_frame: np.ndarray,
                                  src_kp: np.ndarray, src_conf: np.ndarray,
                                  dst_kp: np.ndarray, dst_conf: np.ndarray,
                                  image_size: int = 512) -> np.ndarray:
    """Warp image using per-limb affine transforms.

    Args:
        clothed_frame: Source RGBA image
        src_kp: Source keypoints (18, 2) normalized
        src_conf: Source confidence (18,)
        dst_kp: Destination keypoints (18, 2) normalized
        dst_conf: Destination confidence (18,)
        image_size: Image dimensions

    Returns:
        Warped RGBA image
    """
    # Convert keypoints to pixels
    src_kp_px = src_kp * image_size
    dst_kp_px = dst_kp * image_size

    # Initialize output as zeros
    output = np.zeros_like(clothed_frame, dtype=np.float32)
    total_weights = np.zeros((image_size, image_size), dtype=np.float32)

    # Process each body region
    for region_name, region_indices in BODY_REGIONS.items():
        # Get transform for this region
        M, valid = get_region_transform(src_kp, dst_kp, src_conf, dst_conf,
                                       region_indices, image_size)

        if not valid:
            continue

        # Warp the entire image (all channels at once) using this transform
        warped_region = cv2.warpAffine(
            clothed_frame, M,
            (image_size, image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        ).astype(np.float32)  # Convert to float32 for blending

        # Create weight map for this region
        weights = create_region_weights(image_size, dst_kp_px, dst_conf,
                                       region_indices, sigma=60.0)

        # Blend this region into output (broadcast weights across channels)
        output += warped_region * weights[:, :, np.newaxis]
        total_weights += weights

    # Handle zero-weight pixels by falling back to original image
    low_weight_mask = total_weights < 0.01  # Pixels with insufficient coverage

    # Normalize where we have coverage
    total_weights = np.maximum(total_weights, 1e-6)
    for c in range(output.shape[2]):
        output[:, :, c] /= total_weights

    # Fallback to original for uncovered pixels
    for c in range(output.shape[2]):
        output[:, :, c] = np.where(low_weight_mask, clothed_frame[:, :, c], output[:, :, c])

    return output.astype(np.uint8)


def align_frame_with_skeleton_warping(base_frame_path: str, clothed_frame_path: str) -> np.ndarray:
    """Align clothed frame to base using per-limb skeleton warping.

    Args:
        base_frame_path: Path to base frame
        clothed_frame_path: Path to clothed frame

    Returns:
        Warped and aligned clothing as RGBA numpy array
    """
    print(f"  Extracting skeletons...")

    # Extract keypoints
    base_kp_dict = extract_openpose_keypoints(base_frame_path)
    clothed_kp_dict = extract_openpose_keypoints(clothed_frame_path)

    # Convert to position and confidence arrays
    base_kp, base_conf = get_keypoint_positions(base_kp_dict)
    clothed_kp, clothed_conf = get_keypoint_positions(clothed_kp_dict)

    print(f"  Calculating per-limb warping...")
    print(f"    Base keypoints: {np.sum(base_conf > 0.1)}/18 detected")
    print(f"    Clothed keypoints: {np.sum(clothed_conf > 0.1)}/18 detected")

    # Load clothed image
    clothed_img = np.array(Image.open(clothed_frame_path).convert('RGBA'))

    # Warp clothing to match base skeleton
    warped = warp_with_per_limb_transforms(clothed_img, clothed_kp, clothed_conf,
                                           base_kp, base_conf)

    print(f"  ✓ Skeleton-based warping complete")

    return warped


def main():
    """Align all 25 clothed frames using per-limb skeleton warping."""
    from sprite_clothing_gen.comfy_client import ComfyUIClient

    # Check ComfyUI is running
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        print("ERROR: ComfyUI server not running at http://127.0.0.1:8188")
        print("Start it with: cd /Users/roberthyatt/Code/ComfyUI && python main.py")
        return 1

    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/frames_skeleton_warped")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("SKELETON-GUIDED WARPING (Per-Limb Affine Transforms)")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_idx:02d}.png"

        try:
            # Align using skeleton warping
            aligned = align_frame_with_skeleton_warping(str(base_path), str(clothed_path))

            # Save
            output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
            Image.fromarray(aligned).save(output_path)
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

        print()

    print("=" * 70)
    print("✓ Skeleton-guided warping complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
