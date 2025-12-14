"""Morphological cleanup for mask refinement.

Fills holes and removes islands in segmentation mask using morphological operations.
"""

import cv2
import numpy as np


def cleanup_mask(mask_512: np.ndarray, close_iterations: int = 2, open_iterations: int = 1) -> np.ndarray:
    """Clean up segmentation mask using morphological operations.

    Applies morphological closing to fill small holes in clothing regions,
    then morphological opening to remove isolated base character pixels.

    Args:
        mask_512: 512×512 binary mask (0=base, 1=clothing) as uint8 numpy array
        close_iterations: Number of closing iterations (default: 2, fills holes up to ~4 pixels)
        open_iterations: Number of opening iterations (default: 1, removes islands up to ~2 pixels)

    Returns:
        512×512 cleaned binary mask (0=base, 1=clothing) as uint8 numpy array

    Raises:
        ValueError: If mask_512 is not 512×512
        ValueError: If iterations are not positive
    """
    # Validate inputs
    if mask_512.shape != (512, 512):
        raise ValueError(f"mask_512 must be 512x512, got {mask_512.shape}")
    if close_iterations < 0:
        raise ValueError(f"close_iterations must be non-negative, got {close_iterations}")
    if open_iterations < 0:
        raise ValueError(f"open_iterations must be non-negative, got {open_iterations}")

    # Create 3×3 structuring element
    kernel = np.ones((3, 3), np.uint8)

    # Step 1: CLOSE operation - fills holes in clothing regions
    # Dilation followed by erosion
    mask_closed = cv2.morphologyEx(mask_512, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    # Step 2: OPEN operation - removes isolated base character pixels
    # Erosion followed by dilation
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    return mask_cleaned
