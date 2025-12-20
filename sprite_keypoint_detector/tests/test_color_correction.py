"""Tests for color_correction module."""

import numpy as np
import pytest
from sprite_keypoint_detector.color_correction import remap_frame_to_palette


def test_remap_frame_produces_binary_alpha():
    """After remapping, all pixels should have alpha 0 or 255 only."""
    # Create test frame with semi-transparent pixels
    frame = np.zeros((100, 100, 4), dtype=np.uint8)

    # Add some fully opaque pixels (should become alpha=255)
    frame[20:40, 20:40, :3] = [100, 150, 200]  # BGR
    frame[20:40, 20:40, 3] = 255

    # Add semi-transparent pixels with alpha > 128 (should become alpha=255)
    frame[50:60, 50:60, :3] = [50, 100, 150]
    frame[50:60, 50:60, 3] = 200  # Semi-transparent but above threshold

    # Add semi-transparent pixels with alpha <= 128 (should become alpha=0)
    frame[70:80, 70:80, :3] = [30, 60, 90]
    frame[70:80, 70:80, 3] = 100  # Semi-transparent below threshold

    # Add very low alpha pixels (should become alpha=0)
    frame[85:90, 85:90, :3] = [10, 20, 30]
    frame[85:90, 85:90, 3] = 50

    # Create simple palette
    palette = np.array([
        [100, 150, 200],  # Color 1
        [50, 100, 150],   # Color 2
        [30, 60, 90],     # Color 3
    ], dtype=np.uint8)

    result = remap_frame_to_palette(frame, palette)

    # All alpha values should be either 0 or 255
    unique_alphas = np.unique(result[:, :, 3])
    assert set(unique_alphas).issubset({0, 255}), \
        f"Expected only alpha 0 or 255, got: {unique_alphas}"


def test_remap_preserves_visible_pixel_count_approximately():
    """Remapping should preserve roughly the same visible area."""
    frame = np.zeros((100, 100, 4), dtype=np.uint8)

    # Add visible pixels
    frame[20:80, 20:80, :3] = [100, 150, 200]
    frame[20:80, 20:80, 3] = 255

    palette = np.array([[100, 150, 200]], dtype=np.uint8)

    original_visible = np.sum(frame[:, :, 3] > 128)
    result = remap_frame_to_palette(frame, palette)
    result_visible = np.sum(result[:, :, 3] == 255)

    # Should be exactly the same for fully opaque input
    assert result_visible == original_visible


def test_remap_eliminates_low_alpha_pixels():
    """Pixels with alpha <= 128 should become fully transparent."""
    frame = np.zeros((100, 100, 4), dtype=np.uint8)

    # Add low-alpha pixels (the shadow artifact case)
    frame[40:60, 40:60, :3] = [10, 10, 10]  # Dark shadow color
    frame[40:60, 40:60, 3] = 50  # Low alpha - should be eliminated

    palette = np.array([[10, 10, 10]], dtype=np.uint8)

    result = remap_frame_to_palette(frame, palette)

    # The low-alpha region should now be fully transparent
    assert np.all(result[40:60, 40:60, 3] == 0), \
        "Low alpha pixels should become fully transparent"
