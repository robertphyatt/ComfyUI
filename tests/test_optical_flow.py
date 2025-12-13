import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile


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
