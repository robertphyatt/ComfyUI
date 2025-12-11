#!/usr/bin/env python3
"""Extract clothing spritesheet using corrected masks.

This script uses the manually corrected masks to extract clothing pixels
from the clothed frames and assembles them into a clothing-only spritesheet.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def extract_clothing_spritesheet(
    frames_dir: Path,
    masks_dir: Path,
    output_path: Path,
    grid_size: tuple[int, int] = (5, 5)
):
    """Extract clothing spritesheet using corrected masks.

    Args:
        frames_dir: Directory containing clothed_frame_XX.png files
        masks_dir: Directory containing mask_XX.png files
        output_path: Where to save the clothing spritesheet
        grid_size: Grid dimensions (columns, rows)
    """
    cols, rows = grid_size
    total_frames = cols * rows

    # Load first frame to get dimensions
    first_frame = Image.open(frames_dir / "clothed_frame_00.png").convert('RGBA')
    frame_width, frame_height = first_frame.size

    # Create output spritesheet
    spritesheet_width = frame_width * cols
    spritesheet_height = frame_height * rows
    spritesheet = Image.new('RGBA', (spritesheet_width, spritesheet_height), (0, 0, 0, 0))

    print(f"Extracting clothing from {total_frames} frames...")
    print(f"Output spritesheet: {spritesheet_width}x{spritesheet_height}")

    for frame_idx in range(total_frames):
        # Load clothed frame and mask
        clothed_path = frames_dir / f"clothed_frame_{frame_idx:02d}.png"
        mask_path = masks_dir / f"mask_{frame_idx:02d}.png"

        if not clothed_path.exists():
            print(f"Warning: {clothed_path.name} not found, skipping")
            continue

        if not mask_path.exists():
            print(f"Warning: {mask_path.name} not found, skipping")
            continue

        # Load images
        clothed = np.array(Image.open(clothed_path).convert('RGBA'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Create clothing-only frame
        # Where mask > 128 (white), keep clothed pixels; else transparent
        clothing = clothed.copy()
        clothing[:, :, 3] = np.where(mask > 128, clothed[:, :, 3], 0)

        # Convert back to PIL
        clothing_frame = Image.fromarray(clothing)

        # Calculate position in spritesheet (row-major order)
        row = frame_idx // cols
        col = frame_idx % cols
        x = col * frame_width
        y = row * frame_height

        # Paste into spritesheet
        spritesheet.paste(clothing_frame, (x, y), clothing_frame)

        # Count non-transparent pixels for verification
        pixels_count = np.sum(mask > 128)
        print(f"Frame {frame_idx:02d}: {pixels_count:6d} clothing pixels extracted")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spritesheet.save(output_path)

    print(f"\n✓ Clothing spritesheet saved to: {output_path}")
    print(f"  Resolution: {spritesheet_width}x{spritesheet_height}")
    print(f"  Grid: {cols}x{rows} ({total_frames} frames)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract clothing spritesheet using corrected masks"
    )
    parser.add_argument(
        '--frames-dir',
        type=Path,
        default=Path('training_data/frames'),
        help='Directory containing clothed_frame_XX.png files'
    )
    parser.add_argument(
        '--masks-dir',
        type=Path,
        default=Path('training_data/masks_corrected'),
        help='Directory containing mask_XX.png files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('training_data/clothing_spritesheet.png'),
        help='Output path for clothing spritesheet'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.frames_dir.exists():
        print(f"Error: Frames directory not found: {args.frames_dir}")
        return 1

    if not args.masks_dir.exists():
        print(f"Error: Masks directory not found: {args.masks_dir}")
        return 1

    # Extract clothing spritesheet
    try:
        extract_clothing_spritesheet(
            frames_dir=args.frames_dir,
            masks_dir=args.masks_dir,
            output_path=args.output
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
