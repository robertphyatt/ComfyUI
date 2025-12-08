"""Tests for edge detection functionality."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
import cv2

from edge_detection import detect_clothing_edges


def test_detect_clothing_edges_basic():
    """Test edge detection with visible clothing difference."""
    # Create base frame: solid gray
    base = Image.new('RGB', (512, 512), color='gray')

    # Create clothed frame: gray with white square in center (clothing)
    clothed_arr = np.full((512, 512, 3), 128, dtype=np.uint8)  # Gray background
    clothed_arr[200:300, 200:300] = [255, 255, 255]  # White square (clothing)
    clothed = Image.fromarray(clothed_arr, 'RGB')

    edges = detect_clothing_edges(clothed, base)

    # Verify output shape and type
    assert edges.shape == (512, 512)
    assert edges.dtype == np.uint8
    # Verify edges detected (non-zero values at boundaries)
    assert np.any(edges > 0)


def test_detect_clothing_edges_no_difference():
    """Test edge detection when frames are identical."""
    # Create identical frames
    frame = Image.new('RGB', (512, 512), color='gray')

    edges = detect_clothing_edges(frame, frame)

    # Should return all zeros (no edges)
    assert edges.shape == (512, 512)
    assert np.all(edges == 0)


def test_detect_clothing_edges_invalid_clothed_dimensions():
    """Test validation rejects wrong clothed_frame dimensions."""
    clothed_256 = Image.new('RGB', (256, 256), color='gray')
    base_512 = Image.new('RGB', (512, 512), color='gray')

    with pytest.raises(ValueError, match="clothed_frame must be 512x512"):
        detect_clothing_edges(clothed_256, base_512)


def test_detect_clothing_edges_invalid_base_dimensions():
    """Test validation rejects wrong base_frame dimensions."""
    clothed_512 = Image.new('RGB', (512, 512), color='gray')
    base_256 = Image.new('RGB', (256, 256), color='gray')

    with pytest.raises(ValueError, match="base_frame must be 512x512"):
        detect_clothing_edges(clothed_512, base_256)


def test_detect_clothing_edges_mismatched_dimensions():
    """Test validation catches dimension mismatch."""
    # Both wrong, but different
    clothed_256 = Image.new('RGB', (256, 256), color='gray')
    base_128 = Image.new('RGB', (128, 128), color='gray')

    with pytest.raises(ValueError, match="must be 512x512"):
        detect_clothing_edges(clothed_256, base_128)


def test_detect_clothing_edges_canny_parameters():
    """Test Canny edge detection uses correct parameters."""
    clothed = Image.new('RGB', (512, 512), color='white')
    base = Image.new('RGB', (512, 512), color='gray')

    with patch('cv2.Canny', wraps=cv2.Canny) as mock_canny:
        detect_clothing_edges(clothed, base)

        # Verify Canny was called with correct thresholds
        mock_canny.assert_called_once()
        call_args = mock_canny.call_args
        # Check positional args
        if len(call_args[0]) >= 3:
            # Called with positional args
            assert call_args[0][1] == 50  # threshold1
            assert call_args[0][2] == 150  # threshold2
        else:
            # Called with keyword args
            assert call_args[1].get('threshold1') == 50
            assert call_args[1].get('threshold2') == 150


def test_detect_clothing_edges_dilation_parameters():
    """Test edge dilation uses correct parameters."""
    clothed = Image.new('RGB', (512, 512), color='white')
    base = Image.new('RGB', (512, 512), color='gray')

    with patch('cv2.dilate', wraps=cv2.dilate) as mock_dilate:
        detect_clothing_edges(clothed, base)

        # Verify dilate was called with correct parameters
        mock_dilate.assert_called_once()
        args, kwargs = mock_dilate.call_args
        # args[0] is the edge image
        # args[1] is the kernel
        assert args[1].shape == (3, 3)
        assert np.all(args[1] == 1)
        assert kwargs.get('iterations', 1) == 1
