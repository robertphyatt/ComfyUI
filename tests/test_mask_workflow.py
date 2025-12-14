"""Integration test for complete mask generation and correction workflow."""

import shutil
from pathlib import Path
import pytest
from generate_initial_masks import generate_all_masks


def test_complete_workflow_generates_all_masks(tmp_path):
    """Test that workflow generates 25 initial masks."""
    # Setup test directories
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()

    # Copy test frames
    source_frames = Path("training_data/frames")
    for frame_num in range(25):
        shutil.copy(
            source_frames / f"base_frame_{frame_num:02d}.png",
            frames_dir / f"base_frame_{frame_num:02d}.png"
        )
        shutil.copy(
            source_frames / f"clothed_frame_{frame_num:02d}.png",
            frames_dir / f"clothed_frame_{frame_num:02d}.png"
        )

    # Generate masks
    generate_all_masks(frames_dir, masks_dir)

    # Verify all 25 masks created
    mask_files = list(masks_dir.glob("mask_*.png"))
    assert len(mask_files) == 25

    # Verify mask properties
    from PIL import Image
    mask = Image.open(masks_dir / "mask_00.png")
    assert mask.size == (512, 512)
    assert mask.mode == 'L'
