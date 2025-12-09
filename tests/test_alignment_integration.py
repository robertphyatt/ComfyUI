"""Integration test for full alignment pipeline."""

import pytest
from pathlib import Path
from align_with_openpose import align_frame_with_openpose


@pytest.mark.integration
def test_align_frame_with_openpose():
    """Test full alignment pipeline on real frames."""
    base_path = Path("training_data/frames/base_frame_00.png")
    clothed_path = Path("training_data/frames/clothed_frame_00.png")

    if not base_path.exists() or not clothed_path.exists():
        pytest.skip("Test frames not found")

    # This will only pass if ComfyUI server is running
    try:
        aligned = align_frame_with_openpose(str(base_path), str(clothed_path))

        # Should return aligned image as numpy array
        assert aligned is not None
        assert aligned.shape == (512, 512, 4)  # RGBA
        assert aligned.dtype == 'uint8'

    except RuntimeError as e:
        if "ComfyUI server not running" in str(e):
            pytest.skip("ComfyUI server not running")
        raise
