"""Optical flow warping for sprite clothing transfer."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from typing import Optional, Tuple


def load_image_bgr(path: Path) -> np.ndarray:
    """Load image as BGR numpy array for OpenCV processing."""
    img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def save_image_bgr(arr: np.ndarray, path: Path) -> None:
    """Save BGR numpy array as image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    img.save(path)


def compute_optical_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute dense optical flow from source to target.

    Uses Farneback algorithm to compute pixel displacements.

    Args:
        source: Source BGR image
        target: Target BGR image

    Returns:
        Flow field of shape (H, W, 2) with (dx, dy) displacements
    """
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        source_gray, target_gray,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )

    return flow


def warp_image(source: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp source image using optical flow field.

    Args:
        source: Source BGR image to warp
        flow: Flow field from compute_optical_flow

    Returns:
        Warped BGR image
    """
    h, w = flow.shape[:2]

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply flow displacements to coordinates
    # Flow tells us where pixels moved TO, but remap needs where to sample FROM
    # So we subtract the flow to get the sampling coordinates
    map_x = (x - flow[..., 0]).astype(np.float32)
    map_y = (y - flow[..., 1]).astype(np.float32)

    # Remap (warp) the image
    warped = cv2.remap(
        source, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return warped


def create_body_mask(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Create binary mask where body pixels are (non-white background).

    Args:
        image: BGR image
        threshold: Grayscale value above which is considered background

    Returns:
        Binary mask (255 = body, 0 = background)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < threshold).astype(np.uint8) * 255
    return mask


def blend_with_background(
    warped: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    dilate_iterations: int = 1
) -> np.ndarray:
    """Blend warped armor onto background using mask.

    Args:
        warped: Warped clothed image
        background: Background image (white or mannequin)
        mask: Body mask
        dilate_iterations: How much to dilate mask for edge cleanup

    Returns:
        Blended result
    """
    # Dilate mask to avoid edge artifacts
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    # Normalize mask to 0-1 range
    mask_norm = mask_dilated.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)

    # Blend: warped where mask=1, background where mask=0
    result = (warped * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

    return result


def images_already_aligned(
    clothed: np.ndarray,
    mannequin: np.ndarray,
    threshold: float = 0.98
) -> bool:
    """Check if clothed and mannequin images are already aligned.

    Compares body silhouettes to determine if poses match.
    If they match, no warping is needed.

    Args:
        clothed: BGR clothed image
        mannequin: BGR mannequin image
        threshold: Similarity threshold (0-1), higher = stricter

    Returns:
        True if images are already aligned (no warp needed)
    """
    # Create masks for both images
    clothed_mask = create_body_mask(clothed)
    mannequin_mask = create_body_mask(mannequin)

    # Calculate intersection over union (IoU) of masks
    intersection = np.logical_and(clothed_mask > 0, mannequin_mask > 0).sum()
    union = np.logical_or(clothed_mask > 0, mannequin_mask > 0).sum()

    if union == 0:
        return True  # Both empty = aligned

    iou = intersection / union
    return iou >= threshold


def warp_clothing_to_pose(
    clothed_path: Path,
    mannequin_path: Path,
    output_path: Path,
    debug_dir: Optional[Path] = None,
    alignment_threshold: float = 0.98
) -> Tuple[Path, bool]:
    """Warp clothed reference to match mannequin pose.

    Main entry point for clothing transfer. If images are already
    aligned (poses match), copies the clothed image directly without
    warping to preserve maximum quality.

    Args:
        clothed_path: Path to clothed reference frame
        mannequin_path: Path to base mannequin frame (target pose)
        output_path: Where to save result
        debug_dir: Optional directory for debug outputs
        alignment_threshold: IoU threshold for skip-warp optimization

    Returns:
        Tuple of (output_path, was_skipped) where was_skipped is True
        if the image was already aligned and copied without warping
    """
    # Load images
    clothed = load_image_bgr(clothed_path)
    mannequin = load_image_bgr(mannequin_path)

    # Check if already aligned - skip warp if so
    if images_already_aligned(clothed, mannequin, alignment_threshold):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clothed_path, output_path)
        return output_path, True  # Skipped warping

    # Compute flow: how pixels move from clothed to mannequin
    flow = compute_optical_flow(clothed, mannequin)

    # Warp clothed image to match mannequin pose
    warped = warp_image(clothed, flow)

    # Create mask from mannequin (where body pixels are)
    mask = create_body_mask(mannequin)

    # Blend with clean white background
    white_bg = np.ones_like(mannequin) * 255
    result = blend_with_background(warped, white_bg, mask)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_bgr(result, output_path)

    # Save debug outputs if requested
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_image_bgr(warped, debug_dir / f"warped_{output_path.stem}.png")
        cv2.imwrite(str(debug_dir / f"mask_{output_path.stem}.png"), mask)

    return output_path, False  # Did warp
