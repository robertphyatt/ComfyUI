"""Palette-based color synchronization for sprite animations.

Extracts an optimal N-color palette from all frames using k-means clustering,
then remaps every pixel to the nearest palette color. This ensures all frames
use identical colors regardless of pose differences.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
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
        mask = alpha > 128  # 50% alpha threshold
        bgr = frame[:, :, :3][mask]  # Shape: (N, 3)
        if len(bgr) > 0:
            all_pixels.append(bgr)

    if len(all_pixels) == 0:
        raise ValueError("No visible pixels found in any frame")

    all_pixels = np.vstack(all_pixels)  # Shape: (total_pixels, 3)
    print(f"  Collected {len(all_pixels):,} pixels from {len(frames)} frames")

    # Check if we have enough unique colors
    unique_colors = len(np.unique(all_pixels, axis=0))
    if unique_colors < n_colors:
        print(f"  WARNING: Only {unique_colors} unique colors found, "
              f"but {n_colors} requested. Reducing palette size.")
        n_colors = unique_colors

    # K-means clustering to find optimal palette
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(all_pixels)

    palette = kmeans.cluster_centers_.astype(np.uint8)  # Shape: (n_colors, 3)
    return palette


def remap_frame_to_palette(
    frame: np.ndarray,
    palette: np.ndarray,
    alpha_threshold: int = 128
) -> np.ndarray:
    """Remap all visible pixels in frame to nearest palette color.

    Uses vectorized numpy operations for performance. After remapping colors,
    thresholds alpha to binary (0 or 255) for crisp pixel art.

    Args:
        frame: BGRA image
        palette: Array of shape (n_colors, 3) with BGR values
        alpha_threshold: Pixels with alpha > threshold become opaque (255),
                        pixels with alpha <= threshold become transparent (0)

    Returns:
        Remapped BGRA image with binary alpha (0 or 255 only)
    """
    result = frame.copy()
    alpha = frame[:, :, 3]

    # Determine which pixels will be visible (above threshold)
    visible_mask = alpha > alpha_threshold

    # Extract visible pixels for color remapping: shape (N, 3)
    visible_pixels = frame[visible_mask][:, :3].astype(np.float32)

    if len(visible_pixels) > 0:
        # Compute distances to all palette colors using broadcasting
        # (N, 1, 3) - (1, palette_size, 3) = (N, palette_size, 3)
        palette_float = palette.astype(np.float32)
        distances = np.sqrt(np.sum(
            (visible_pixels[:, np.newaxis, :] - palette_float[np.newaxis, :, :]) ** 2,
            axis=2
        ))

        # Find nearest palette color for each pixel: shape (N,)
        nearest_indices = np.argmin(distances, axis=1)

        # Remap to palette colors
        result[visible_mask, :3] = palette[nearest_indices]

    # Threshold alpha to binary: visible pixels become 255, others become 0
    result[:, :, 3] = np.where(visible_mask, 255, 0).astype(np.uint8)

    return result


def quantize_frame(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Quantize a frame to use only palette colors with binary alpha.

    This is an alias for remap_frame_to_palette, used for clarity
    when quantizing early in the pipeline vs final cleanup.

    After quantization:
    - All visible pixels (alpha > 128) use exact palette colors with alpha=255
    - All other pixels become fully transparent (alpha=0)
    - No semi-transparent pixels remain

    Args:
        frame: BGRA image
        palette: Array of shape (n_colors, 3) with BGR values

    Returns:
        Quantized BGRA image using only palette colors, binary alpha
    """
    return remap_frame_to_palette(frame, palette)


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


def load_palette_from_image(path: Path) -> np.ndarray:
    """Load palette from a swatch image.

    Parses a 4x4 grid of 32x32 color swatches back to palette array.

    Args:
        path: Path to palette swatch image (128x128 PNG)

    Returns:
        Palette array of shape (16, 3) with BGR values

    Raises:
        FileNotFoundError: If palette image doesn't exist
        ValueError: If image format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Palette not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read palette image: {path}")

    if img.shape[0] < 128 or img.shape[1] < 128:
        raise ValueError(f"Palette image too small: {img.shape}, expected at least 128x128")

    # Sample center of each 32x32 swatch cell
    swatch_size = 32
    palette = []
    for row in range(4):
        for col in range(4):
            center_y = row * swatch_size + swatch_size // 2
            center_x = col * swatch_size + swatch_size // 2
            color = img[center_y, center_x]
            palette.append(color)

    return np.array(palette, dtype=np.uint8)
