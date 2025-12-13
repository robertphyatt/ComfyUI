import pytest
import numpy as np
from pathlib import Path
from PIL import Image


class TestImageIO:
    def test_load_image_bgr_returns_numpy_array(self, tmp_path):
        """Load image should return BGR numpy array."""
        from sprite_clothing_gen.optical_flow import load_image_bgr

        # Create a simple test image (red pixel)
        img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        result = load_image_bgr(img_path)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)
        # BGR format: red = (0, 0, 255)
        assert result[0, 0, 2] == 255  # Red channel
        assert result[0, 0, 0] == 0    # Blue channel

    def test_save_image_bgr_creates_file(self, tmp_path):
        """Save image should create PNG file from BGR array."""
        from sprite_clothing_gen.optical_flow import save_image_bgr

        # Create BGR array (blue pixel in BGR = (255, 0, 0))
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 0]  # Blue in BGR

        output_path = tmp_path / "output.png"
        save_image_bgr(arr, output_path)

        assert output_path.exists()

        # Verify it saved correctly (load and check)
        loaded = Image.open(output_path)
        assert loaded.size == (10, 10)
        # Should be blue (0, 0, 255) in RGB
        assert loaded.getpixel((0, 0)) == (0, 0, 255)
