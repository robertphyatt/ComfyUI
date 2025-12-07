"""Utilities for splitting and reassembling spritesheets."""

from pathlib import Path
from typing import List, Tuple
from PIL import Image


def split_spritesheet(
    spritesheet_path: Path,
    output_dir: Path,
    grid_size: Tuple[int, int] = (5, 5)
) -> List[Path]:
    """Split a spritesheet into individual frame images.

    Args:
        spritesheet_path: Path to input spritesheet
        output_dir: Directory to save individual frames
        grid_size: Tuple of (columns, rows) in the grid

    Returns:
        List of paths to individual frame files

    Raises:
        ValueError: If spritesheet dimensions don't match grid size
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(spritesheet_path)
    width, height = img.size

    cols, rows = grid_size
    frame_width = width // cols
    frame_height = height // rows

    # Verify dimensions divide evenly
    if width % cols != 0 or height % rows != 0:
        raise ValueError(
            f"Spritesheet size {width}x{height} doesn't divide evenly "
            f"into {cols}x{rows} grid"
        )

    frame_paths = []
    frame_index = 0

    for row in range(rows):
        for col in range(cols):
            # Calculate crop box for this frame
            left = col * frame_width
            top = row * frame_height
            right = left + frame_width
            bottom = top + frame_height

            # Extract frame
            frame = img.crop((left, top, right, bottom))

            # Save frame
            frame_path = output_dir / f"frame_{frame_index:02d}.png"
            frame.save(frame_path)
            frame_paths.append(frame_path)

            frame_index += 1

    return frame_paths


def reassemble_spritesheet(
    frame_paths: List[Path],
    output_path: Path,
    grid_size: Tuple[int, int] = (5, 5)
) -> Path:
    """Reassemble individual frames into a spritesheet.

    Args:
        frame_paths: List of paths to frame images (must be in order)
        output_path: Path to save output spritesheet
        grid_size: Tuple of (columns, rows) in the grid

    Returns:
        Path to output spritesheet

    Raises:
        ValueError: If number of frames doesn't match grid size
    """
    cols, rows = grid_size
    expected_frames = cols * rows

    if len(frame_paths) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frames for {cols}x{rows} grid, "
            f"got {len(frame_paths)}"
        )

    # Load first frame to get dimensions
    first_frame = Image.open(frame_paths[0])
    frame_width, frame_height = first_frame.size

    # Create output image
    sheet_width = frame_width * cols
    sheet_height = frame_height * rows
    spritesheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))

    # Paste frames into grid
    frame_index = 0
    for row in range(rows):
        for col in range(cols):
            frame = Image.open(frame_paths[frame_index])

            # Calculate position
            x = col * frame_width
            y = row * frame_height

            # Paste frame
            spritesheet.paste(frame, (x, y))
            frame_index += 1

    # Save spritesheet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spritesheet.save(output_path)

    return output_path
