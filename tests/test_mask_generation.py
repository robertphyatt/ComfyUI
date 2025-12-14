"""Tests for inpainting mask generation."""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from generate_inpainting_masks import generate_mask_from_alpha


def test_mask_from_alpha_creates_binary_mask():
    """Test that alpha channel creates proper binary mask."""
    # Create test RGBA image (100x100 with center square opaque)
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[25:75, 25:75, 3] = 255  # Opaque square in center

    test_img = Image.fromarray(img, 'RGBA')

    # Generate mask
    mask = generate_mask_from_alpha(test_img)

    # Verify mask is binary (0 or 255)
    assert mask.mode == 'L'
    assert mask.size == (100, 100)

    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)
    assert len(unique_values) <= 2
    assert 0 in unique_values
    assert 255 in unique_values

    # Verify mask matches alpha channel
    assert np.all(mask_array[25:75, 25:75] == 255)  # Center is white
    assert np.all(mask_array[0:25, :] == 0)  # Top is black


def test_mask_generation_for_all_base_frames():
    """Test generating masks for actual base frames."""
    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/masks_inpainting")

    base_path = frames_dir / "base_frame_00.png"
    if not base_path.exists():
        pytest.skip("Base frame not found")

    from generate_inpainting_masks import generate_masks_for_frames

    # Generate for frame 0 only
    generate_masks_for_frames(frames_dir, output_dir, frame_range=(0, 1))

    # Verify output exists
    mask_path = output_dir / "mask_00.png"
    assert mask_path.exists()

    # Load and verify
    mask = Image.open(mask_path)
    assert mask.mode == 'L'
    assert mask.size == (512, 512)
