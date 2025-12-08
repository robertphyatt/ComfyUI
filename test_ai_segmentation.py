# test_ai_segmentation.py
import numpy as np
import pytest
import requests
from PIL import Image
from unittest.mock import patch, MagicMock
from ai_segmentation import call_ollama_segmentation

def test_call_ollama_segmentation_basic():
    """Test AI segmentation calls Ollama and decodes RLE response."""
    # Create 256x256 test image
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    # Mock Ollama response with run-length encoding
    mock_response = {
        "response": '{"mask": [{"value": 1, "count": 65536}]}'  # All clothing
    }

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )

        mask = call_ollama_segmentation(clothed_256)

        # Verify shape and content
        assert mask.shape == (256, 256)
        assert np.all(mask == 1)  # All pixels classified as clothing

def test_call_ollama_segmentation_mixed():
    """Test AI segmentation handles mixed clothing/base regions."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    # Mock response: first 32k pixels = base (0), rest = clothing (1)
    mock_response = {
        "response": '{"mask": [{"value": 0, "count": 32768}, {"value": 1, "count": 32768}]}'
    }

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )

        mask = call_ollama_segmentation(clothed_256)

        # First half should be 0, second half should be 1
        assert mask.shape == (256, 256)
        flat = mask.flatten()
        assert np.all(flat[:32768] == 0)
        assert np.all(flat[32768:] == 1)

def test_call_ollama_segmentation_network_error():
    """Test network failure handling."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    with patch('requests.post', side_effect=requests.RequestException("Connection failed")):
        with pytest.raises(RuntimeError, match="Failed to call Ollama"):
            call_ollama_segmentation(clothed_256)

def test_call_ollama_segmentation_malformed_json():
    """Test AI returns invalid JSON."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    mock_response = {"response": "{invalid json here}"}

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)

        with pytest.raises(RuntimeError, match="Failed to parse Ollama response"):
            call_ollama_segmentation(clothed_256)

def test_call_ollama_segmentation_missing_mask_key():
    """Test AI returns valid JSON with wrong schema."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    mock_response = {"response": '{"result": [{"value": 1, "count": 65536}]}'}

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)

        with pytest.raises(RuntimeError, match="Failed to parse Ollama response"):
            call_ollama_segmentation(clothed_256)

def test_call_ollama_segmentation_rle_sum_mismatch():
    """Test RLE validation catches sum mismatch."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    # Only 30,000 pixels instead of 65,536
    mock_response = {"response": '{"mask": [{"value": 1, "count": 30000}]}'}

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)

        with pytest.raises(ValueError, match="RLE sum"):
            call_ollama_segmentation(clothed_256)

def test_call_ollama_segmentation_invalid_rle_values():
    """Test RLE validation rejects values other than 0/1."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    # Invalid value: 2 (should be 0 or 1)
    mock_response = {"response": '{"mask": [{"value": 2, "count": 65536}]}'}

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)

        with pytest.raises(ValueError, match="RLE values must be 0 or 1"):
            call_ollama_segmentation(clothed_256)
