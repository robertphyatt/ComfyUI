"""Edge detection for clothing boundary identification.

Detects edges in the difference between clothed and base frames to identify
precise clothing boundaries for mask refinement.
"""

import cv2
import numpy as np
from PIL import Image


def detect_clothing_edges(clothed_frame: Image.Image, base_frame: Image.Image) -> np.ndarray:
    """Detect edges in difference between clothed and base frames.

    Applies Canny edge detection to the absolute difference between frames,
    then dilates to create forgiving search zones for boundary snapping.

    Args:
        clothed_frame: PIL Image (512×512) of character wearing clothing
        base_frame: PIL Image (512×512) of base character without clothing

    Returns:
        512×512 binary edge map (255=edge, 0=no edge) as uint8 numpy array

    Raises:
        ValueError: If images are not 512×512 or don't match dimensions
    """
    # Validate dimensions
    if clothed_frame.size != (512, 512):
        raise ValueError(f"clothed_frame must be 512x512, got {clothed_frame.size}")
    if base_frame.size != (512, 512):
        raise ValueError(f"base_frame must be 512x512, got {base_frame.size}")
    if clothed_frame.size != base_frame.size:
        raise ValueError(f"Frame dimensions must match: {clothed_frame.size} vs {base_frame.size}")

    # Convert to grayscale numpy arrays
    clothed_gray = np.array(clothed_frame.convert('L'))
    base_gray = np.array(base_frame.convert('L'))

    # Compute absolute difference
    diff = cv2.absdiff(clothed_gray, base_gray)

    # Detect edges with Canny
    edges = cv2.Canny(diff, threshold1=50, threshold2=150)

    # Dilate edges to create search zones (2-3 pixel width)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    return edges_dilated
