"""Integration test for the clothing pipeline."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json

from ..transform import transform_frame, TransformConfig, get_keypoints_array
from ..spritesheet import split_spritesheet, assemble_spritesheet, detect_layout
from ..matching import compute_joint_distance, find_top_candidates
from ..validation import validate_frame, compute_median_bone_lengths


class TestTransformPipeline:
    """Test the transform pipeline components."""

    def test_transform_produces_output(self):
        """Transform should produce non-empty output."""
        # Create simple test images
        clothed = np.zeros((512, 512, 4), dtype=np.uint8)
        clothed[200:300, 200:300, :3] = 128  # Gray square
        clothed[200:300, 200:300, 3] = 255   # Opaque

        base = np.zeros((512, 512, 4), dtype=np.uint8)
        base[210:310, 210:310, :3] = 100
        base[210:310, 210:310, 3] = 255

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[200:300, 200:300] = 255

        # Simple keypoints (just need neck for alignment)
        clothed_kpts = np.zeros((18, 2))
        clothed_kpts[1] = [250, 200]  # neck
        clothed_kpts[10] = [240, 300]  # left hip
        clothed_kpts[11] = [260, 300]  # right hip

        base_kpts = np.zeros((18, 2))
        base_kpts[1] = [260, 210]  # neck (offset)
        base_kpts[10] = [250, 310]
        base_kpts[11] = [270, 310]

        config = TransformConfig(scale_factor=1.0, pixelize_factor=1)

        result = transform_frame(clothed, clothed_kpts, base, base_kpts, mask, config)

        assert result is not None
        assert result.shape == (512, 512, 4)
        assert np.any(result[:, :, 3] > 0)  # Has some visible pixels


class TestSpritesheet:
    """Test spritesheet utilities."""

    def test_split_and_assemble_roundtrip(self):
        """Split then assemble should produce same image."""
        # Create test spritesheet (2x2 grid of 64x64 frames)
        sheet = np.zeros((128, 128, 4), dtype=np.uint8)
        sheet[0:64, 0:64, 0] = 255    # Red frame
        sheet[0:64, 64:128, 1] = 255  # Green frame
        sheet[64:128, 0:64, 2] = 255  # Blue frame
        sheet[64:128, 64:128, :3] = 128  # Gray frame
        sheet[:, :, 3] = 255  # All opaque

        from ..spritesheet import SpritesheetLayout
        layout = SpritesheetLayout(
            frame_width=64, frame_height=64,
            columns=2, rows=2, total_frames=4
        )

        frames = split_spritesheet(sheet, layout)
        assert len(frames) == 4

        reassembled = assemble_spritesheet(frames, layout)
        assert np.array_equal(sheet, reassembled)


class TestMatching:
    """Test frame matching."""

    def test_joint_distance_identical(self):
        """Identical keypoints should have zero distance."""
        kpts = {"head": [100, 100], "neck": [100, 150]}
        dist = compute_joint_distance(kpts, kpts, ["head", "neck"])
        assert dist == 0.0

    def test_joint_distance_different(self):
        """Different keypoints should have positive distance."""
        kpts1 = {"head": [100, 100], "neck": [100, 150]}
        kpts2 = {"head": [110, 100], "neck": [100, 160]}
        dist = compute_joint_distance(kpts1, kpts2, ["head", "neck"])
        assert dist > 0


class TestValidation:
    """Test annotation validation."""

    def test_validate_good_frame(self):
        """Valid keypoints should pass validation."""
        # Provide all 18 keypoints with valid positions
        kpts = {
            "head": {"x": 256, "y": 100, "source": "manual", "confidence": 1.0},
            "neck": {"x": 256, "y": 150, "source": "manual", "confidence": 1.0},
            "left_shoulder": {"x": 200, "y": 170, "source": "manual", "confidence": 1.0},
            "right_shoulder": {"x": 312, "y": 170, "source": "manual", "confidence": 1.0},
            "left_elbow": {"x": 180, "y": 220, "source": "manual", "confidence": 1.0},
            "right_elbow": {"x": 332, "y": 220, "source": "manual", "confidence": 1.0},
            "left_wrist": {"x": 170, "y": 270, "source": "manual", "confidence": 1.0},
            "right_wrist": {"x": 342, "y": 270, "source": "manual", "confidence": 1.0},
            "left_fingertip": {"x": 165, "y": 300, "source": "manual", "confidence": 1.0},
            "right_fingertip": {"x": 347, "y": 300, "source": "manual", "confidence": 1.0},
            "left_hip": {"x": 230, "y": 280, "source": "manual", "confidence": 1.0},
            "right_hip": {"x": 282, "y": 280, "source": "manual", "confidence": 1.0},
            "left_knee": {"x": 225, "y": 350, "source": "manual", "confidence": 1.0},
            "right_knee": {"x": 287, "y": 350, "source": "manual", "confidence": 1.0},
            "left_ankle": {"x": 220, "y": 420, "source": "manual", "confidence": 1.0},
            "right_ankle": {"x": 292, "y": 420, "source": "manual", "confidence": 1.0},
            "left_toe": {"x": 215, "y": 450, "source": "manual", "confidence": 1.0},
            "right_toe": {"x": 297, "y": 450, "source": "manual", "confidence": 1.0},
        }
        result = validate_frame("test.png", kpts)
        assert result.is_valid

    def test_validate_out_of_bounds(self):
        """Out of bounds keypoints should fail."""
        kpts = {
            "head": {"x": 600, "y": 100, "source": "auto", "confidence": 0.9},
        }
        result = validate_frame("test.png", kpts, image_bounds=(512, 512))
        assert not result.is_valid
        assert any("outside bounds" in issue for issue in result.issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
