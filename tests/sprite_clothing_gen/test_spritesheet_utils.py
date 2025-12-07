"""Tests for spritesheet utilities."""

import pytest
from pathlib import Path
from PIL import Image
from sprite_clothing_gen.spritesheet_utils import split_spritesheet, reassemble_spritesheet


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_spritesheet(temp_dir):
    """Create a sample 5x5 spritesheet for testing."""
    # Create 5x5 grid of 64x64 frames = 320x320 image
    frame_size = 64
    grid_size = (5, 5)
    sheet_width = frame_size * grid_size[0]
    sheet_height = frame_size * grid_size[1]

    # Create image with different color per frame for easy identification
    img = Image.new('RGBA', (sheet_width, sheet_height))
    pixels = img.load()

    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            # Calculate color based on position
            r = (row * 50) % 256
            g = (col * 50) % 256
            b = ((row + col) * 30) % 256

            # Fill frame area with color
            for y in range(frame_size):
                for x in range(frame_size):
                    px = col * frame_size + x
                    py = row * frame_size + y
                    pixels[px, py] = (r, g, b, 255)

    sheet_path = temp_dir / "test_spritesheet.png"
    img.save(sheet_path)
    return sheet_path


def test_split_spritesheet(sample_spritesheet, temp_dir):
    """Test splitting spritesheet into individual frames."""
    output_dir = temp_dir / "frames"
    output_dir.mkdir()

    frames = split_spritesheet(sample_spritesheet, output_dir, grid_size=(5, 5))

    # Should produce 25 frames
    assert len(frames) == 25

    # All frames should exist
    for frame_path in frames:
        assert frame_path.exists()
        assert frame_path.suffix == '.png'

    # Frames should be named frame_00.png through frame_24.png
    assert frames[0].name == "frame_00.png"
    assert frames[24].name == "frame_24.png"

    # Each frame should be 64x64
    img = Image.open(frames[0])
    assert img.size == (64, 64)


def test_reassemble_spritesheet(temp_dir):
    """Test reassembling frames into spritesheet."""
    # Create 25 test frames (5x5 grid)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir()

    frame_paths = []
    frame_size = 64

    for i in range(25):
        # Create frame with unique color
        r = (i * 10) % 256
        g = (i * 15) % 256
        b = (i * 20) % 256

        img = Image.new('RGBA', (frame_size, frame_size), (r, g, b, 255))
        frame_path = frames_dir / f"frame_{i:02d}.png"
        img.save(frame_path)
        frame_paths.append(frame_path)

    output_path = temp_dir / "reassembled.png"
    result = reassemble_spritesheet(frame_paths, output_path, grid_size=(5, 5))

    # Output should exist
    assert result.exists()

    # Should be 320x320 (5x5 grid of 64x64 frames)
    img = Image.open(result)
    assert img.size == (320, 320)

    # Verify first frame's color is preserved in top-left
    pixels = img.load()
    assert pixels[0, 0][:3] == (0, 0, 0)  # frame_00's color
