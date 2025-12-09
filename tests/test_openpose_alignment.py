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

    # Base frame keypoints (person centered at 100, 100)
    base_kp = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,  # nose
                100, 100, 0.9,  # neck
                90, 110, 0.9,  # right shoulder
                0, 0, 0, 0, 0, 0,  # elbow, wrist
                110, 110, 0.9,  # left shoulder
            ] + [0] * 42
        }]
    }

    # Clothed frame keypoints (person offset by +10, +5)
    clothed_kp = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,
                110, 105, 0.9,  # neck at 110, 105
                100, 115, 0.9,  # right shoulder at 100, 115
                0, 0, 0, 0, 0, 0,
                120, 115, 0.9,  # left shoulder at 120, 115
            ] + [0] * 42
        }]
    }

    offset_x, offset_y = calculate_alignment_offset(base_kp, clothed_kp)

    # Base center: (100+90+110)/3 = 100, (100+110+110)/3 ≈ 106.67
    # Clothed center: (110+100+120)/3 = 110, (105+115+115)/3 ≈ 111.67
    # Offset: 100 - 110 = -10, 106.67 - 111.67 = -5
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
