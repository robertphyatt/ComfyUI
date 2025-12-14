"""Tests for morphological cleanup functionality."""

import numpy as np
import pytest
from unittest.mock import patch
import cv2

from morphological_cleanup import cleanup_mask


def test_cleanup_mask_fills_holes():
    """Test that close operation fills small holes in clothing regions."""
    # Create mask with a small hole in clothing region
    mask_512 = np.ones((512, 512), dtype=np.uint8)
    # Create 2×2 hole in the middle (should be filled by close operation)
    mask_512[255:257, 255:257] = 0

    result = cleanup_mask(mask_512, close_iterations=2, open_iterations=0)

    # Hole should be filled
    assert result.shape == (512, 512)
    assert result.dtype == np.uint8
    # The 2×2 hole should now be filled (all 1s)
    assert np.all(result[255:257, 255:257] == 1), "Small holes should be filled"


def test_cleanup_mask_removes_islands():
    """Test that open operation removes isolated pixels."""
    # Create mask with isolated pixels
    mask_512 = np.zeros((512, 512), dtype=np.uint8)
    # Create isolated 1×1 pixel (should be removed by open operation)
    mask_512[256, 256] = 1

    result = cleanup_mask(mask_512, close_iterations=0, open_iterations=1)

    # Isolated pixel should be removed
    assert result.shape == (512, 512)
    assert result[256, 256] == 0, "Isolated pixels should be removed"


def test_cleanup_mask_combined():
    """Test combined close and open operations."""
    # Create mask with both holes and islands
    mask_512 = np.ones((512, 512), dtype=np.uint8)
    # Small hole (should be filled)
    mask_512[100:102, 100:102] = 0

    # Large region of base character with isolated clothing pixels
    mask_512[200:300, 200:300] = 0
    mask_512[250, 250] = 1  # Isolated pixel (should be removed)

    result = cleanup_mask(mask_512, close_iterations=2, open_iterations=1)

    # Verify shape and type
    assert result.shape == (512, 512)
    assert result.dtype == np.uint8
    # Small hole should be filled
    assert np.all(result[100:102, 100:102] == 1)
    # Isolated pixel should be removed
    assert result[250, 250] == 0


def test_cleanup_mask_no_iterations():
    """Test that mask is unchanged when iterations=0."""
    mask_512 = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

    result = cleanup_mask(mask_512, close_iterations=0, open_iterations=0)

    # Should return unchanged mask
    assert np.array_equal(result, mask_512)


def test_cleanup_mask_invalid_dimensions():
    """Test validation rejects wrong mask dimensions."""
    mask_256 = np.zeros((256, 256), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask_512 must be 512x512"):
        cleanup_mask(mask_256)


def test_cleanup_mask_invalid_close_iterations():
    """Test validation rejects negative close_iterations."""
    mask_512 = np.zeros((512, 512), dtype=np.uint8)

    with pytest.raises(ValueError, match="close_iterations must be non-negative"):
        cleanup_mask(mask_512, close_iterations=-1)


def test_cleanup_mask_invalid_open_iterations():
    """Test validation rejects negative open_iterations."""
    mask_512 = np.zeros((512, 512), dtype=np.uint8)

    with pytest.raises(ValueError, match="open_iterations must be non-negative"):
        cleanup_mask(mask_512, open_iterations=-1)


def test_cleanup_mask_morph_close_parameters():
    """Test that MORPH_CLOSE is called with correct parameters."""
    mask_512 = np.zeros((512, 512), dtype=np.uint8)

    with patch('cv2.morphologyEx', wraps=cv2.morphologyEx) as mock_morph:
        cleanup_mask(mask_512, close_iterations=2, open_iterations=1)

        # First call should be MORPH_CLOSE
        first_call = mock_morph.call_args_list[0]
        args, kwargs = first_call
        assert args[1] == cv2.MORPH_CLOSE
        assert args[2].shape == (3, 3)  # 3×3 kernel
        assert kwargs.get('iterations') == 2


def test_cleanup_mask_morph_open_parameters():
    """Test that MORPH_OPEN is called with correct parameters."""
    mask_512 = np.zeros((512, 512), dtype=np.uint8)

    with patch('cv2.morphologyEx', wraps=cv2.morphologyEx) as mock_morph:
        cleanup_mask(mask_512, close_iterations=2, open_iterations=1)

        # Second call should be MORPH_OPEN
        second_call = mock_morph.call_args_list[1]
        args, kwargs = second_call
        assert args[1] == cv2.MORPH_OPEN
        assert args[2].shape == (3, 3)  # 3×3 kernel
        assert kwargs.get('iterations') == 1


def test_cleanup_mask_operation_order():
    """Test that operations happen in correct order (close, then open)."""
    mask_512 = np.zeros((512, 512), dtype=np.uint8)

    with patch('cv2.morphologyEx', wraps=cv2.morphologyEx) as mock_morph:
        cleanup_mask(mask_512, close_iterations=2, open_iterations=1)

        # Should be called exactly twice
        assert mock_morph.call_count == 2

        # First call: MORPH_CLOSE
        assert mock_morph.call_args_list[0][0][1] == cv2.MORPH_CLOSE

        # Second call: MORPH_OPEN
        assert mock_morph.call_args_list[1][0][1] == cv2.MORPH_OPEN
