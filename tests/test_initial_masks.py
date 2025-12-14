# tests/test_initial_masks.py
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
from generate_initial_masks import generate_mask_from_color_diff


def test_generate_mask_identifies_brown_armor_pixels():
    """Test that color-based mask generation identifies brown armor pixels."""
    # Create test images
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    base[200:300, 200:300] = [128, 128, 128]  # Gray head region

    clothed = base.copy()
    clothed[250:400, 150:350] = [101, 67, 33]  # Brown armor overlapping head

    mask = generate_mask_from_color_diff(base, clothed)

    # Verify mask shape
    assert mask.shape == (512, 512)
    assert mask.dtype == np.uint8

    # Verify armor region marked as 1
    assert mask[300, 250] == 1  # Below head, should be armor

    # Verify background marked as 0
    assert mask[100, 100] == 0  # Far from character
