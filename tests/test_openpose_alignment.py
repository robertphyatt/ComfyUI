"""Tests for OpenPose skeleton-based alignment."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from align_with_openpose import extract_openpose_keypoints, parse_keypoints


def test_extract_openpose_keypoints_returns_valid_structure():
    """Test that OpenPose extraction returns keypoints in expected format."""
    # Use existing test frame
    frame_path = Path("training_data/frames/base_frame_00.png")

    if not frame_path.exists():
        pytest.skip("Test frame not found")

    try:
        keypoints = extract_openpose_keypoints(str(frame_path))
    except RuntimeError as e:
        if "ComfyUI server not running" in str(e):
            pytest.skip("ComfyUI server not running")
        raise

    # Should return dict with 'people' array
    assert keypoints is not None
    assert 'people' in keypoints
    assert len(keypoints['people']) > 0

    # First person should have pose_keypoints_2d
    person = keypoints['people'][0]
    assert 'pose_keypoints_2d' in person

    # Should have 18 keypoints * 3 values (x, y, confidence)
    assert len(person['pose_keypoints_2d']) == 54


def test_parse_keypoints_extracts_upper_body_center():
    """Test that we correctly extract upper body center from keypoints."""
    # Mock keypoints for a person
    mock_keypoints = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,  # 0: nose
                100, 100, 0.9,  # 1: neck
                90, 110, 0.9,  # 2: right shoulder
                0, 0, 0,  # 3: right elbow
                0, 0, 0,  # 4: right wrist
                110, 110, 0.9,  # 5: left shoulder
            ] + [0] * 48  # Remaining keypoints
        }]
    }

    center_x, center_y = parse_keypoints(mock_keypoints)

    # Should average neck (100, 100), right shoulder (90, 110), left shoulder (110, 110)
    # Expected: x = (100 + 90 + 110) / 3 = 100, y = (100 + 110 + 110) / 3 ≈ 106.67
    assert abs(center_x - 100.0) < 0.1
    assert abs(center_y - 106.67) < 0.1


def test_calculate_alignment_offset():
    """Test calculating alignment offset from two sets of keypoints."""
    from align_with_openpose import calculate_alignment_offset

    # OpenPose returns normalized coordinates (0.0 to 1.0)
    # Base frame keypoints (person centered at normalized 0.2, 0.2 = pixel 102.4, 102.4)
    base_kp = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,  # nose
                0.2, 0.2, 0.9,  # neck
                0.18, 0.22, 0.9,  # right shoulder
                0, 0, 0, 0, 0, 0,  # elbow, wrist
                0.22, 0.22, 0.9,  # left shoulder
            ] + [0] * 42
        }]
    }

    # Clothed frame keypoints (offset by +0.02, +0.01 normalized = +10.24, +5.12 pixels)
    clothed_kp = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,
                0.22, 0.21, 0.9,  # neck at 0.22, 0.21
                0.20, 0.23, 0.9,  # right shoulder at 0.20, 0.23
                0, 0, 0, 0, 0, 0,
                0.24, 0.23, 0.9,  # left shoulder at 0.24, 0.23
            ] + [0] * 42
        }]
    }

    offset_x, offset_y = calculate_alignment_offset(base_kp, clothed_kp)

    # Base center: (0.2+0.18+0.22)/3 = 0.2, (0.2+0.22+0.22)/3 ≈ 0.2133
    # Clothed center: (0.22+0.20+0.24)/3 = 0.22, (0.21+0.23+0.23)/3 ≈ 0.2233
    # In pixels (×512): Base=(102.4, 109.2), Clothed=(112.6, 114.3)
    # Offset: 102.4 - 112.6 ≈ -10, 109.2 - 114.3 ≈ -5
    assert offset_x == -10
    assert offset_y == -5


def test_apply_alignment_transform():
    """Test applying alignment offset to shift an image."""
    from align_with_openpose import apply_alignment_transform

    # Create test image: 100x100 white square in top-left corner
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    img[0:100, 0:100] = [255, 255, 255, 255]  # White square

    # Shift by +50, +50
    offset_x, offset_y = 50, 50

    aligned = apply_alignment_transform(img, offset_x, offset_y)

    # White square should now be at 50:150, 50:150
    assert np.all(aligned[50:150, 50:150] == [255, 255, 255, 255])
    # Original position should be transparent
    assert np.all(aligned[0:50, 0:50, 3] == 0)
