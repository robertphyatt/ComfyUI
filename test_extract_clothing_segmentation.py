"""Tests for semantic segmentation-based clothing extraction."""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

from extract_clothing_segmentation import extract_clothing_semantic


def test_extract_clothing_semantic_basic():
    """Test basic extraction pipeline with mocked components."""
    clothed = Image.new('RGB', (512, 512), color='brown')
    base = Image.new('RGB', (512, 512), color='gray')

    # Mock AI segmentation to return a simple mask
    mock_mask_128 = np.ones((128, 128), dtype=np.uint8)

    # Mock edge detection to return edges
    mock_edges_512 = np.zeros((512, 512), dtype=np.uint8)

    with patch('extract_clothing_segmentation.call_ollama_segmentation', return_value=mock_mask_128):
        with patch('extract_clothing_segmentation.detect_clothing_edges', return_value=mock_edges_512):
            result = extract_clothing_semantic(clothed, base)

    # Verify output
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)
    assert result.mode == 'RGBA'


def test_extract_clothing_semantic_pipeline_integration():
    """Test that all pipeline steps are called in correct order."""
    clothed = Image.new('RGB', (512, 512), color='brown')
    base = Image.new('RGB', (512, 512), color='gray')

    # Create mock objects to track call order
    mock_mask_128 = np.ones((128, 128), dtype=np.uint8)
    mock_edges_512 = np.zeros((512, 512), dtype=np.uint8)
    mock_mask_512_snapped = np.ones((512, 512), dtype=np.uint8)
    mock_mask_512_cleaned = np.ones((512, 512), dtype=np.uint8)

    with patch('extract_clothing_segmentation.call_ollama_segmentation', return_value=mock_mask_128) as mock_ai:
        with patch('extract_clothing_segmentation.detect_clothing_edges', return_value=mock_edges_512) as mock_edges:
            with patch('extract_clothing_segmentation.snap_mask_to_edges', return_value=mock_mask_512_snapped) as mock_snap:
                with patch('extract_clothing_segmentation.cleanup_mask', return_value=mock_mask_512_cleaned) as mock_cleanup:
                    result = extract_clothing_semantic(clothed, base)

    # Verify all functions were called
    mock_ai.assert_called_once()
    mock_edges.assert_called_once_with(clothed, base)
    mock_snap.assert_called_once_with(mock_mask_128, mock_edges_512, search_radius=10)
    mock_cleanup.assert_called_once_with(mock_mask_512_snapped, close_iterations=2, open_iterations=1)

    # Verify result
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGBA'


def test_extract_clothing_semantic_downscaling():
    """Test that clothed frame is downscaled to 128×128 for AI."""
    clothed = Image.new('RGB', (512, 512), color='brown')
    base = Image.new('RGB', (512, 512), color='gray')

    mock_mask_128 = np.ones((128, 128), dtype=np.uint8)

    with patch('extract_clothing_segmentation.call_ollama_segmentation', return_value=mock_mask_128) as mock_ai:
        with patch('extract_clothing_segmentation.detect_clothing_edges', return_value=np.zeros((512, 512), dtype=np.uint8)):
            extract_clothing_semantic(clothed, base)

    # Verify AI was called with 128×128 image
    call_args = mock_ai.call_args[0]
    downscaled_image = call_args[0]
    assert downscaled_image.size == (128, 128)


def test_extract_clothing_semantic_transparency():
    """Test that output has transparent background (alpha=0 where mask=0)."""
    clothed = Image.new('RGB', (512, 512), color='brown')
    base = Image.new('RGB', (512, 512), color='gray')

    # Mock mask: top half is clothing (1), bottom half is base (0)
    mock_mask_cleaned = np.zeros((512, 512), dtype=np.uint8)
    mock_mask_cleaned[0:256, :] = 1  # Top half is clothing

    with patch('extract_clothing_segmentation.call_ollama_segmentation', return_value=np.ones((128, 128), dtype=np.uint8)):
        with patch('extract_clothing_segmentation.detect_clothing_edges', return_value=np.zeros((512, 512), dtype=np.uint8)):
            with patch('extract_clothing_segmentation.cleanup_mask', return_value=mock_mask_cleaned):
                result = extract_clothing_semantic(clothed, base)

    # Convert to array to check alpha channel
    result_arr = np.array(result)

    # Top half should have alpha > 0 (clothing visible)
    assert np.any(result_arr[0:256, :, 3] > 0), "Clothing region should be visible"

    # Bottom half should have alpha = 0 (base removed)
    assert np.all(result_arr[256:512, :, 3] == 0), "Base character region should be transparent"


def test_extract_clothing_semantic_invalid_clothed_dimensions():
    """Test validation rejects wrong clothed_frame dimensions."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')
    base_512 = Image.new('RGB', (512, 512), color='gray')

    with pytest.raises(ValueError, match="clothed_frame must be 512x512"):
        extract_clothing_semantic(clothed_256, base_512)


def test_extract_clothing_semantic_invalid_base_dimensions():
    """Test validation rejects wrong base_frame dimensions."""
    clothed_512 = Image.new('RGB', (512, 512), color='brown')
    base_256 = Image.new('RGB', (256, 256), color='gray')

    with pytest.raises(ValueError, match="base_frame must be 512x512"):
        extract_clothing_semantic(clothed_512, base_256)


def test_extract_clothing_semantic_mismatched_dimensions():
    """Test validation catches dimension mismatch."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')
    base_128 = Image.new('RGB', (128, 128), color='gray')

    with pytest.raises(ValueError, match="must be 512x512"):
        extract_clothing_semantic(clothed_256, base_128)
