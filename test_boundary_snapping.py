"""Tests for boundary snapping functionality."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

from boundary_snapping import snap_mask_to_edges


def test_snap_mask_to_edges_basic():
    """Test basic boundary snapping with nearby edges."""
    # Create simple mask: 128×128 block of clothing in center
    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    mask_256[64:192, 64:192] = 1

    # Create edge map at 512×512 with edges around expected upscaled boundary
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    # Draw rectangle of edges (slightly offset from perfect upscaling)
    edges_512[130:380, 130:380] = 255  # Offset by ~2 pixels from 128,128 to 384,384

    result = snap_mask_to_edges(mask_256, edges_512, search_radius=10)

    # Verify output shape and type
    assert result.shape == (512, 512)
    assert result.dtype == np.uint8
    # Verify binary values
    assert np.all((result == 0) | (result == 1))


def test_snap_mask_to_edges_no_edges():
    """Test snapping with no edges detected (empty edge map)."""
    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    mask_256[64:192, 64:192] = 1

    edges_512 = np.zeros((512, 512), dtype=np.uint8)  # No edges

    result = snap_mask_to_edges(mask_256, edges_512)

    # Should return upscaled mask unchanged (no edges to snap to)
    expected = cv2.resize(mask_256, (512, 512), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(result, expected)


def test_snap_mask_to_edges_edges_too_far():
    """Test no snapping when edges are beyond search_radius."""
    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    mask_256[64:192, 64:192] = 1

    # Create edges far from boundaries (> 10 pixels away)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    edges_512[0:50, :] = 255  # Edges far from mask boundaries

    result = snap_mask_to_edges(mask_256, edges_512, search_radius=10)

    # Should return upscaled mask unchanged (edges too far)
    expected = cv2.resize(mask_256, (512, 512), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(result, expected)


def test_snap_mask_to_edges_invalid_mask_dimensions():
    """Test validation rejects wrong mask dimensions."""
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask_256 must be 256x256"):
        snap_mask_to_edges(mask_128, edges_512)


def test_snap_mask_to_edges_invalid_edge_dimensions():
    """Test validation rejects wrong edge map dimensions."""
    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    edges_256 = np.zeros((256, 256), dtype=np.uint8)

    with pytest.raises(ValueError, match="edges_512 must be 512x512"):
        snap_mask_to_edges(mask_256, edges_256)


def test_snap_mask_to_edges_invalid_search_radius():
    """Test validation rejects invalid search_radius."""
    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with pytest.raises(ValueError, match="search_radius must be positive"):
        snap_mask_to_edges(mask_256, edges_512, search_radius=0)

    with pytest.raises(ValueError, match="search_radius must be positive"):
        snap_mask_to_edges(mask_256, edges_512, search_radius=-5)


def test_snap_mask_to_edges_upscaling():
    """Test mask upscaling uses INTER_NEAREST."""
    mask_256 = np.zeros((256, 256), dtype=np.uint8)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with patch('cv2.resize', wraps=cv2.resize) as mock_resize:
        snap_mask_to_edges(mask_256, edges_512)

        # Verify resize was called with INTER_NEAREST
        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        assert args[1] == (512, 512)  # Target size
        assert kwargs.get('interpolation') == cv2.INTER_NEAREST
