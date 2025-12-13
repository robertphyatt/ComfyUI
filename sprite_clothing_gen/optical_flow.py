"""Optical flow warping for sprite clothing transfer."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


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
