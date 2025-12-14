"""Tests for clothing extraction using U2-Net."""

import pytest
from pathlib import Path
from PIL import Image
from sprite_clothing_gen.clothing_extractor import extract_clothing_from_reference


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_clothed_character(temp_dir):
    """Create a sample image of a clothed character."""
    # Create simple test image: blue body, red clothing
    img = Image.new('RGBA', (64, 64), (0, 0, 255, 255))  # Blue background (body)
    pixels = img.load()

    # Add red rectangle in center to represent clothing
    for y in range(20, 44):
        for x in range(20, 44):
            pixels[x, y] = (255, 0, 0, 255)  # Red (clothing)

    img_path = temp_dir / "clothed_character.png"
    img.save(img_path)
    return img_path


def test_extract_clothing_basic(sample_clothed_character, temp_dir):
    """Test basic clothing extraction returns valid image."""
    output_path = temp_dir / "clothing_only.png"

    result = extract_clothing_from_reference(
        sample_clothed_character,
        output_path,
        model="u2net"  # Use basic model for testing
    )

    # Output should exist
    assert result.exists()

    # Should be PNG with alpha channel
    img = Image.open(result)
    assert img.mode == 'RGBA'
    assert img.size == (64, 64)
