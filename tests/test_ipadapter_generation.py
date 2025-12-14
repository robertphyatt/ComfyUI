"""Tests for IPAdapter generation script."""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from generate_with_ipadapter import generate_clothed_frame


def test_generate_clothed_frame_validates_paths():
    """Test that generation fails when base frame is missing."""
    client = Mock()
    frames_dir = Path("nonexistent")
    masks_dir = Path("nonexistent")
    output_dir = Path("output")

    result = generate_clothed_frame(client, 0, frames_dir, masks_dir, output_dir)

    assert result is False
    client.upload_image.assert_not_called()


def test_generate_clothed_frame_uploads_images(tmp_path):
    """Test that generation uploads base image and mask."""
    # Create test files
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    base_path = frames_dir / "base_frame_00.png"
    mask_path = masks_dir / "mask_00.png"

    # Create minimal PNG files
    from PIL import Image
    img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    img.save(base_path)
    img.save(mask_path)

    # Mock client
    client = Mock()
    client.queue_prompt.return_value = "test_prompt_id"
    client.wait_for_completion.return_value = {
        'outputs': {
            '1': {
                'images': [{
                    'filename': 'test_output.png',
                    'subfolder': ''
                }]
            }
        }
    }
    client.download_image.return_value = output_dir / "test_output.png"

    # Create the output file so rename works
    output_file = output_dir / "test_output.png"
    img.save(output_file)

    result = generate_clothed_frame(client, 0, frames_dir, masks_dir, output_dir)

    assert result is True
    assert client.upload_image.call_count == 2
    client.queue_prompt.assert_called_once()
    client.wait_for_completion.assert_called_once()


def test_generate_clothed_frame_handles_generation_failure(tmp_path):
    """Test that generation handles ComfyUI errors gracefully."""
    # Create test files
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    base_path = frames_dir / "base_frame_00.png"
    mask_path = masks_dir / "mask_00.png"

    from PIL import Image
    img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    img.save(base_path)
    img.save(mask_path)

    # Mock client that fails
    client = Mock()
    client.queue_prompt.return_value = "test_prompt_id"
    client.wait_for_completion.side_effect = RuntimeError("Generation timeout")

    result = generate_clothed_frame(client, 0, frames_dir, masks_dir, output_dir)

    assert result is False
