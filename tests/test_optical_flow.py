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


class TestMasking:
    def test_create_body_mask_detects_non_white(self):
        """Body mask should be 255 where non-white, 0 where white."""
        from sprite_clothing_gen.optical_flow import create_body_mask

        # Create image with white background and gray body
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255  # All white
        img[20:30, 20:30] = [100, 100, 100]  # Gray square (body)

        mask = create_body_mask(img)

        assert mask.shape == (50, 50)
        assert mask[25, 25] == 255  # Body region
        assert mask[0, 0] == 0      # Background

    def test_blend_with_background_uses_mask(self):
        """Blend should use warped where mask=255, background where mask=0."""
        from sprite_clothing_gen.optical_flow import blend_with_background

        # Warped image: all red
        warped = np.zeros((50, 50, 3), dtype=np.uint8)
        warped[:, :] = [0, 0, 255]  # Red in BGR

        # Background: all blue
        background = np.zeros((50, 50, 3), dtype=np.uint8)
        background[:, :] = [255, 0, 0]  # Blue in BGR

        # Mask: center square is body
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255

        result = blend_with_background(warped, background, mask)

        # Center should be red (from warped)
        assert result[25, 25, 2] > 200  # Red channel high
        # Edge should be blue (from background)
        assert result[0, 0, 0] > 200    # Blue channel high


class TestWarpClothingToPose:
    def test_warp_clothing_to_pose_creates_output(self, tmp_path):
        """Main function should create output file when poses differ."""
        from sprite_clothing_gen.optical_flow import warp_clothing_to_pose

        # Create simple clothed image (gray body on white)
        clothed = Image.new('RGB', (50, 50), color=(255, 255, 255))
        for x in range(20, 30):
            for y in range(20, 30):
                clothed.putpixel((x, y), (100, 80, 60))  # Brown-ish
        clothed_path = tmp_path / "clothed.png"
        clothed.save(clothed_path)

        # Create mannequin image (DIFFERENT pose - shifted)
        mannequin = Image.new('RGB', (50, 50), color=(255, 255, 255))
        for x in range(25, 35):  # Shifted +5 pixels
            for y in range(25, 35):
                mannequin.putpixel((x, y), (128, 128, 128))  # Gray
        mannequin_path = tmp_path / "mannequin.png"
        mannequin.save(mannequin_path)

        output_path = tmp_path / "output.png"

        result_path, was_skipped = warp_clothing_to_pose(clothed_path, mannequin_path, output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert was_skipped == False  # Should have warped

    def test_warp_clothing_to_pose_skips_when_aligned(self, tmp_path):
        """Should skip warping and copy directly when poses already match."""
        from sprite_clothing_gen.optical_flow import warp_clothing_to_pose

        # Create clothed and mannequin with SAME pose
        clothed = Image.new('RGB', (50, 50), color=(255, 255, 255))
        for x in range(20, 30):
            for y in range(20, 30):
                clothed.putpixel((x, y), (100, 80, 60))  # Brown armor
        clothed_path = tmp_path / "clothed.png"
        clothed.save(clothed_path)

        # Mannequin at SAME position
        mannequin = Image.new('RGB', (50, 50), color=(255, 255, 255))
        for x in range(20, 30):  # Same position as clothed
            for y in range(20, 30):
                mannequin.putpixel((x, y), (128, 128, 128))  # Gray
        mannequin_path = tmp_path / "mannequin.png"
        mannequin.save(mannequin_path)

        output_path = tmp_path / "output.png"

        result_path, was_skipped = warp_clothing_to_pose(clothed_path, mannequin_path, output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert was_skipped == True  # Should have skipped warping

        # Verify output is identical to clothed (direct copy)
        output_img = Image.open(output_path)
        clothed_img = Image.open(clothed_path)
        assert list(output_img.getdata()) == list(clothed_img.getdata())


class TestAlphaBasedAlignment:
    def test_images_already_aligned_detects_alpha_mismatch(self, tmp_path):
        """Alignment check should fail when alpha channels differ significantly."""
        from sprite_clothing_gen.optical_flow import images_already_aligned_alpha
        from PIL import Image
        import numpy as np

        # Create base image with full body visible (alpha=255 for body area)
        base = Image.new('RGBA', (100, 100), (255, 255, 255, 0))  # Transparent bg
        base_arr = np.array(base)
        base_arr[20:80, 30:70, :3] = [128, 128, 128]  # Gray body
        base_arr[20:80, 30:70, 3] = 255  # Visible
        base_path = tmp_path / "base.png"
        Image.fromarray(base_arr).save(base_path)

        # Create clothed image with SMALLER visible area (shoulders missing)
        clothed = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
        clothed_arr = np.array(clothed)
        clothed_arr[30:80, 30:70, :3] = [139, 69, 19]  # Brown armor (smaller)
        clothed_arr[30:80, 30:70, 3] = 255  # Visible
        clothed_path = tmp_path / "clothed.png"
        Image.fromarray(clothed_arr).save(clothed_path)

        # Should NOT be aligned - base has visible pixels clothed doesn't cover
        result = images_already_aligned_alpha(clothed_path, base_path, threshold=0.98)
        assert result == False, "Should detect alpha mismatch (mannequin showing through)"
