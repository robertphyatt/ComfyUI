#!/usr/bin/env python3
"""Test semantic segmentation extraction on a single frame.

Usage:
    python3 test_segmentation_single_frame.py [frame_number]

Examples:
    python3 test_segmentation_single_frame.py        # Test frame 0
    python3 test_segmentation_single_frame.py 5      # Test frame 5
"""

import sys
import time
from pathlib import Path
from PIL import Image

from extract_clothing_segmentation import extract_clothing_semantic


def extract_frame_from_spritesheet(spritesheet: Image.Image, frame_num: int,
                                   frames_per_row: int = 5, frame_size: int = 512) -> Image.Image:
    """Extract a single frame from a spritesheet.

    Args:
        spritesheet: PIL Image containing all frames
        frame_num: Frame number (0-24)
        frames_per_row: Number of frames per row (default: 5)
        frame_size: Size of each frame in pixels (default: 512)

    Returns:
        PIL Image of the extracted frame (512×512)
    """
    row = frame_num // frames_per_row
    col = frame_num % frames_per_row

    left = col * frame_size
    top = row * frame_size
    right = left + frame_size
    bottom = top + frame_size

    return spritesheet.crop((left, top, right, bottom))


def main():
    # Parse command line arguments
    frame_num = 0
    if len(sys.argv) > 1:
        try:
            frame_num = int(sys.argv[1])
            if not (0 <= frame_num <= 24):
                print(f"Error: Frame number must be 0-24, got {frame_num}", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid frame number '{sys.argv[1]}'", file=sys.stderr)
            print("Usage: python3 test_segmentation_single_frame.py [frame_number]", file=sys.stderr)
            sys.exit(1)

    # Define paths
    base_sheet_path = Path("examples/input/base.png")
    clothed_sheet_path = Path("examples/input/reference.png")
    output_dir = Path("debug_frames_semantic")
    output_path = output_dir / f"frame_{frame_num:02d}_clothing.png"

    # Validate input files exist
    if not base_sheet_path.exists():
        print(f"Error: Base spritesheet not found: {base_sheet_path}", file=sys.stderr)
        sys.exit(1)
    if not clothed_sheet_path.exists():
        print(f"Error: Clothed spritesheet not found: {clothed_sheet_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing semantic segmentation on frame {frame_num}")
    print(f"  Base sheet: {base_sheet_path}")
    print(f"  Clothed sheet: {clothed_sheet_path}")
    print(f"  Output: {output_path}")
    print()

    # Load sprite sheets
    print("Loading sprite sheets...")
    base_sheet = Image.open(base_sheet_path)
    clothed_sheet = Image.open(clothed_sheet_path)

    # Extract specific frame
    print(f"Extracting frame {frame_num}...")
    base_frame = extract_frame_from_spritesheet(base_sheet, frame_num)
    clothed_frame = extract_frame_from_spritesheet(clothed_sheet, frame_num)

    # Run extraction
    print("Running semantic segmentation extraction...")
    print()
    start_time = time.time()

    try:
        clothing_only = extract_clothing_semantic(clothed_frame, base_frame)

        elapsed = time.time() - start_time
        print()
        print(f"✓ Extraction completed in {elapsed:.2f}s")

        # Save output
        print(f"Saving to {output_path}...")
        clothing_only.save(output_path)

        print(f"✓ Success! Output saved to {output_path}")
        print()
        print("To view the result:")
        print(f"  open {output_path}")

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print(f"✗ Extraction failed after {elapsed:.2f}s", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
