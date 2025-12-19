"""Palette-based color synchronization for sprite animations.

Extracts an optimal N-color palette from all frames using k-means clustering,
then remaps every pixel to the nearest palette color. This ensures all frames
use identical colors regardless of pose differences.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.cluster import KMeans


def extract_palette(frames: List[np.ndarray], n_colors: int = 16) -> np.ndarray:
    """Extract optimal n-color palette from all frames using k-means.

    Args:
        frames: List of BGRA images
        n_colors: Number of colors in palette (default 16 for SNES-style)

    Returns:
        Palette array of shape (n_colors, 3) with BGR values
    """
    # Collect all visible pixels from all frames
    all_pixels = []
    for frame in frames:
        alpha = frame[:, :, 3]
        mask = alpha > 128
        bgr = frame[:, :, :3][mask]  # Shape: (N, 3)
        all_pixels.append(bgr)

    all_pixels = np.vstack(all_pixels)  # Shape: (total_pixels, 3)
    print(f"  Collected {len(all_pixels):,} pixels from {len(frames)} frames")

    # K-means clustering to find optimal palette
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(all_pixels)

    palette = kmeans.cluster_centers_.astype(np.uint8)  # Shape: (n_colors, 3)
    return palette


def remap_frame_to_palette(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Remap all visible pixels in frame to nearest palette color.

    Args:
        frame: BGRA image
        palette: Array of shape (n_colors, 3) with BGR values

    Returns:
        Remapped BGRA image
    """
    result = frame.copy()
    alpha = frame[:, :, 3]
    mask = alpha > 128

    # Get visible pixel coordinates
    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        pixel_bgr = frame[y, x, :3].astype(np.float32)

        # Find nearest palette color (Euclidean distance)
        distances = np.sqrt(np.sum((palette.astype(np.float32) - pixel_bgr) ** 2, axis=1))
        nearest_idx = np.argmin(distances)

        result[y, x, :3] = palette[nearest_idx]

    return result


def remap_all_frames(
    frames: List[np.ndarray],
    palette: np.ndarray
) -> List[np.ndarray]:
    """Remap all frames to use the shared palette.

    Args:
        frames: List of BGRA images
        palette: Array of shape (n_colors, 3) with BGR values

    Returns:
        List of remapped BGRA images
    """
    results = []
    for i, frame in enumerate(frames):
        remapped = remap_frame_to_palette(frame, palette)
        results.append(remapped)
        print(f"  Frame {i:02d}: remapped to palette")
    return results


def save_palette_image(palette: np.ndarray, path: Path) -> None:
    """Save palette as a visual swatch image.

    Creates a 4x4 grid of 32x32 color swatches.

    Args:
        palette: Array of shape (n_colors, 3) with BGR values
        path: Output path for the image
    """
    import cv2

    n_colors = len(palette)
    cols = 4
    rows = (n_colors + cols - 1) // cols
    swatch_size = 32

    img = np.zeros((rows * swatch_size, cols * swatch_size, 3), dtype=np.uint8)

    for i, color in enumerate(palette):
        row, col = i // cols, i % cols
        y1, y2 = row * swatch_size, (row + 1) * swatch_size
        x1, x2 = col * swatch_size, (col + 1) * swatch_size
        img[y1:y2, x1:x2] = color

    cv2.imwrite(str(path), img)
