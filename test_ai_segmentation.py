# test_ai_segmentation.py
import numpy as np
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
