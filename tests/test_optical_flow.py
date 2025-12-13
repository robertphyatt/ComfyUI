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


class TestOpticalFlow:
    def test_compute_optical_flow_returns_flow_field(self):
        """Optical flow should return (H, W, 2) displacement field."""
        from sprite_clothing_gen.optical_flow import compute_optical_flow

        # Create two simple images: source and shifted target
        source = np.zeros((50, 50, 3), dtype=np.uint8)
        source[20:30, 20:30] = [255, 255, 255]  # White square

        target = np.zeros((50, 50, 3), dtype=np.uint8)
        target[22:32, 22:32] = [255, 255, 255]  # Shifted +2 pixels

        flow = compute_optical_flow(source, target)

        assert isinstance(flow, np.ndarray)
        assert flow.shape == (50, 50, 2)  # (H, W, 2) for dx, dy
        assert flow.dtype == np.float32

    def test_warp_image_applies_flow(self):
        """Warp should move pixels according to flow field."""
        from sprite_clothing_gen.optical_flow import warp_image

        # Create source image with white square
        source = np.zeros((50, 50, 3), dtype=np.uint8)
        source[20:30, 20:30] = [255, 255, 255]

        # Create flow that shifts everything +5 pixels in x
        flow = np.zeros((50, 50, 2), dtype=np.float32)
        flow[:, :, 0] = 5  # dx = 5

        warped = warp_image(source, flow)

        assert warped.shape == source.shape
        # Original white square was at x=20:30
        # After +5 shift, it should be at x=25:35
        # Check center of shifted region has white pixels
        assert warped[25, 30, 0] > 200  # Should be white-ish
