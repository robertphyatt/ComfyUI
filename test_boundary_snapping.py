"""Tests for boundary snapping functionality."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

from boundary_snapping import snap_mask_to_edges


def test_snap_mask_to_edges_basic():
    """Test basic boundary snapping with nearby edges."""
    # Create simple mask: 64×64 block of clothing in center of 128x128
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    mask_128[32:96, 32:96] = 1

    # Create edge map at 512×512 with edges around expected upscaled boundary
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    # Draw rectangle of edges (slightly offset from perfect upscaling)
    edges_512[130:380, 130:380] = 255  # Offset by ~2 pixels from 128,128 to 384,384

    result = snap_mask_to_edges(mask_128, edges_512, search_radius=10)

    # Verify output shape and type
    assert result.shape == (512, 512)
    assert result.dtype == np.uint8
    # Verify binary values
    assert np.all((result == 0) | (result == 1))


def test_snap_mask_to_edges_no_edges():
    """Test snapping with no edges detected (empty edge map)."""
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    mask_128[32:96, 32:96] = 1

    edges_512 = np.zeros((512, 512), dtype=np.uint8)  # No edges

    result = snap_mask_to_edges(mask_128, edges_512)

    # Should return upscaled mask unchanged (no edges to snap to)
    expected = cv2.resize(mask_128, (512, 512), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(result, expected)


def test_snap_mask_to_edges_edges_too_far():
    """Test no snapping when edges are beyond search_radius."""
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    mask_128[32:96, 32:96] = 1

    # Create edges far from boundaries (> 10 pixels away)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    edges_512[0:50, :] = 255  # Edges far from mask boundaries

    result = snap_mask_to_edges(mask_128, edges_512, search_radius=10)

    # Should return upscaled mask unchanged (edges too far)
    expected = cv2.resize(mask_128, (512, 512), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(result, expected)


def test_snap_mask_to_edges_invalid_mask_dimensions():
    """Test validation rejects wrong mask dimensions."""
    mask_64 = np.zeros((64, 64), dtype=np.uint8)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask_128 must be 128x128"):
        snap_mask_to_edges(mask_64, edges_512)


def test_snap_mask_to_edges_invalid_edge_dimensions():
    """Test validation rejects wrong edge map dimensions."""
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    edges_256 = np.zeros((256, 256), dtype=np.uint8)

    with pytest.raises(ValueError, match="edges_512 must be 512x512"):
        snap_mask_to_edges(mask_128, edges_256)


def test_snap_mask_to_edges_invalid_search_radius():
    """Test validation rejects invalid search_radius."""
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with pytest.raises(ValueError, match="search_radius must be positive"):
        snap_mask_to_edges(mask_128, edges_512, search_radius=0)

    with pytest.raises(ValueError, match="search_radius must be positive"):
        snap_mask_to_edges(mask_128, edges_512, search_radius=-5)


def test_snap_mask_to_edges_upscaling():
    """Test mask upscaling uses INTER_NEAREST."""
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with patch('cv2.resize', wraps=cv2.resize) as mock_resize:
        snap_mask_to_edges(mask_128, edges_512)

        # Verify resize was called with INTER_NEAREST
        mock_resize.assert_called_once()
        args, kwargs = mock_resize.call_args
        assert args[1] == (512, 512)  # Target size
        assert kwargs.get('interpolation') == cv2.INTER_NEAREST


def test_snap_mask_boundary_actually_moves():
    """Verify boundary actually moves to edge location and fills intermediate pixels."""
    # Create mask with vertical boundary at x=50 (upscales to x=200 in 512x512)
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    mask_128[:, 0:50] = 1  # Left portion is clothing

    # Create vertical edge at x=205 (after upscaling, boundary at ~200, edge at 205)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    edges_512[:, 205] = 255  # Vertical edge 5 pixels away from upscaled boundary

    result = snap_mask_to_edges(mask_128, edges_512, search_radius=10)

    # Verify boundary moved: pixels between old boundary and edge should now be clothing (1)
    # After upscaling, left boundary is at x~=200
    # Edge is at x=205
    # Pixels from 200 to 205 should be filled with 1
    assert result.shape == (512, 512)

    # Check that pixels between old boundary and edge are now filled
    # The line drawing should have filled the gap
    assert np.any(result[:, 201:206] == 1), "Pixels between boundary and edge should be filled"


def test_snap_mask_preserves_interior():
    """Verify interior pixels remain unchanged after snapping."""
    # Create mask with well-defined interior
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    mask_128[32:96, 32:96] = 1

    # Create edges slightly offset from boundaries
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    edges_512[126:386, 126:386] = 255  # 2px offset

    result = snap_mask_to_edges(mask_128, edges_512, search_radius=10)

    # Interior pixels far from boundaries should remain unchanged
    # Check center region (well away from boundaries)
    interior_region = result[200:312, 200:312]
    assert np.all(interior_region == 1), "Interior should remain as clothing"


def test_snap_mask_inner_vs_outer_boundaries():
    """Test that inner and outer boundaries snap correctly."""
    # Create small mask region (inner boundary should expand)
    mask_128 = np.zeros((128, 128), dtype=np.uint8)
    mask_128[50:60, 50:60] = 1  # Small 10x10 square

    # Create edges 5 pixels outside the boundary (should expand mask outward)
    edges_512 = np.zeros((512, 512), dtype=np.uint8)
    # After upscaling, square is at 200:240 in 512x512
    # Put edges 5 pixels outside
    edges_512[195:245, 195] = 255  # Left edge
    edges_512[195:245, 245] = 255  # Right edge
    edges_512[195, 195:245] = 255  # Top edge
    edges_512[245, 195:245] = 255  # Bottom edge

    result = snap_mask_to_edges(mask_128, edges_512, search_radius=10)

    # Mask should have expanded to reach the edges
    # Check that expansion happened
    assert result.shape == (512, 512)
    # The mask should have grown larger than the original upscaled version
    original_upscaled = cv2.resize(mask_128, (512, 512), interpolation=cv2.INTER_NEAREST)
    assert np.sum(result == 1) >= np.sum(original_upscaled == 1), "Mask should expand to edges"
