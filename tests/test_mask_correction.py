# tests/test_mask_correction.py
import numpy as np
import pytest
from mask_correction_tool import MaskEditor


def test_mask_editor_initializes_with_images():
    """Test that MaskEditor loads images correctly."""
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    clothed = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)

    editor = MaskEditor(base, clothed, mask)

    assert editor.base_img.shape == (512, 512, 3)
    assert editor.clothed_img.shape == (512, 512, 3)
    assert editor.mask.shape == (512, 512)
    assert editor.brush_size > 0


def test_mask_editor_paint_adds_pixels():
    """Test that painting adds pixels to mask."""
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    clothed = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)

    editor = MaskEditor(base, clothed, mask)

    # Paint at position (100, 100) with brush size 5
    editor.paint_at(100, 100, value=1, brush_size=5)

    # Verify pixels were painted in editor's mask
    assert editor.mask[100, 100] == 1
    assert editor.mask[102, 102] == 1  # Within brush radius
