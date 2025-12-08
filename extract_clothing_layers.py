#!/usr/bin/env python3
"""Extract clothing layers by comparing base and clothed spritesheets.

This tool:
1. Splits both spritesheets into 25 frames
2. Aligns each clothed frame to match the base frame position
3. Extracts clothing-only by removing the base character
4. Reassembles into a clothing-only spritesheet with transparent background
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image


def split_spritesheet(spritesheet_path: Path, grid_size: Tuple[int, int] = (5, 5)) -> List[Image.Image]:
    """Split a spritesheet into individual frames.

    Args:
        spritesheet_path: Path to spritesheet image
        grid_size: Grid dimensions (columns, rows)

    Returns:
        List of PIL Images, one per frame
    """
    img = Image.open(spritesheet_path)
    width, height = img.size
    cols, rows = grid_size

    frame_width = width // cols
    frame_height = height // rows

    frames = []
    for row in range(rows):
        for col in range(cols):
            left = col * frame_width
            top = row * frame_height
            right = left + frame_width
            bottom = top + frame_height

            frame = img.crop((left, top, right, bottom))
            frames.append(frame)

    return frames


def find_character_bounds(frame: Image.Image) -> Tuple[int, int, int, int]:
    """Find bounding box of non-transparent pixels in frame.

    Args:
        frame: PIL Image with alpha channel

    Returns:
        (left, top, right, bottom) bounding box, or None if empty
    """
    # Convert to RGBA if needed
    if frame.mode != 'RGBA':
        frame = frame.convert('RGBA')

    # Get alpha channel
    alpha = np.array(frame)[:, :, 3]

    # Find non-zero (non-transparent) pixels
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    return (left, top, right, bottom)


def align_frames(base_frame: Image.Image, clothed_frame: Image.Image) -> Image.Image:
    """Align clothed frame to match base frame position.

    Finds the character in both frames and shifts the clothed frame
    so the character aligns with the base frame.

    Args:
        base_frame: Base character frame
        clothed_frame: Clothed character frame to align

    Returns:
        Aligned clothed frame
    """
    # Find character bounds in both frames
    base_bounds = find_character_bounds(base_frame)
    clothed_bounds = find_character_bounds(clothed_frame)

    # If either frame is empty, return clothed as-is
    if base_bounds is None or clothed_bounds is None:
        return clothed_frame

    base_left, base_top, base_right, base_bottom = base_bounds
    clothed_left, clothed_top, clothed_right, clothed_bottom = clothed_bounds

    # Calculate offset to align centers
    base_center_x = (base_left + base_right) // 2
    base_center_y = (base_top + base_bottom) // 2
    clothed_center_x = (clothed_left + clothed_right) // 2
    clothed_center_y = (clothed_top + clothed_bottom) // 2

    offset_x = base_center_x - clothed_center_x
    offset_y = base_center_y - clothed_center_y

    # Create new image with same size
    aligned = Image.new('RGBA', clothed_frame.size, (0, 0, 0, 0))

    # Paste clothed frame at offset
    aligned.paste(clothed_frame, (offset_x, offset_y), clothed_frame)

    return aligned


def extract_clothing_layer(base_frame: Image.Image, clothed_frame: Image.Image,
                           tolerance: int = 30) -> Image.Image:
    """Extract clothing layer by removing base character from clothed frame.

    Args:
        base_frame: Base character frame
        clothed_frame: Aligned clothed character frame
        tolerance: Color difference tolerance for matching pixels

    Returns:
        Clothing-only frame with transparent background
    """
    # Ensure both are RGBA
    if base_frame.mode != 'RGBA':
        base_frame = base_frame.convert('RGBA')
    if clothed_frame.mode != 'RGBA':
        clothed_frame = clothed_frame.convert('RGBA')

    # Convert to numpy arrays
    base_arr = np.array(base_frame)
    clothed_arr = np.array(clothed_frame)

    # Create output array (start with clothed)
    clothing_arr = clothed_arr.copy()

    # For each pixel, if it matches the base character, make it transparent
    for y in range(base_arr.shape[0]):
        for x in range(base_arr.shape[1]):
            base_pixel = base_arr[y, x]
            clothed_pixel = clothed_arr[y, x]

            # Skip if base pixel is already transparent
            if base_pixel[3] == 0:
                continue

            # Skip if clothed pixel is transparent
            if clothed_pixel[3] == 0:
                continue

            # Calculate color difference (RGB only)
            diff = np.abs(base_pixel[:3].astype(int) - clothed_pixel[:3].astype(int))
            max_diff = np.max(diff)

            # If colors are similar, this is body (not clothing) - make transparent
            if max_diff <= tolerance:
                clothing_arr[y, x] = [0, 0, 0, 0]

    return Image.fromarray(clothing_arr, 'RGBA')


def reassemble_spritesheet(frames: List[Image.Image], output_path: Path,
                          grid_size: Tuple[int, int] = (5, 5)) -> Path:
    """Reassemble frames into a spritesheet.

    Args:
        frames: List of frame images
        output_path: Path to save output spritesheet
        grid_size: Grid dimensions (columns, rows)

    Returns:
        Path to output file
    """
    cols, rows = grid_size
    frame_width = frames[0].width
    frame_height = frames[0].height

    # Create output image
    output = Image.new('RGBA', (frame_width * cols, frame_height * rows), (0, 0, 0, 0))

    # Paste each frame
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * frame_width
        y = row * frame_height
        output.paste(frame, (x, y), frame)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract clothing layers from spritesheet comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python extract_clothing_layers.py \\
        --base examples/input/base.png \\
        --clothed examples/input/reference.png \\
        --output examples/output/clothing.png
        """
    )

    parser.add_argument(
        '--base',
        type=Path,
        required=True,
        help='Base spritesheet (naked character)'
    )

    parser.add_argument(
        '--clothed',
        type=Path,
        required=True,
        help='Clothed spritesheet (character wearing clothing)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output/clothing.png'),
        help='Output path for clothing-only spritesheet (default: output/clothing.png)'
    )

    parser.add_argument(
        '--tolerance',
        type=int,
        default=30,
        help='Color matching tolerance (0-255, default: 30)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save intermediate frames for debugging'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.base.exists():
        print(f"Error: Base spritesheet not found: {args.base}", file=sys.stderr)
        return 1

    if not args.clothed.exists():
        print(f"Error: Clothed spritesheet not found: {args.clothed}", file=sys.stderr)
        return 1

    print("🎨 Extracting clothing layers...")
    print(f"   Base: {args.base}")
    print(f"   Clothed: {args.clothed}")
    print(f"   Tolerance: {args.tolerance}")

    # Step 1: Split spritesheets
    print("\n📦 Step 1: Splitting spritesheets...")
    base_frames = split_spritesheet(args.base)
    clothed_frames = split_spritesheet(args.clothed)
    print(f"   Split into {len(base_frames)} frames each")

    # Step 2: Process each frame
    print("\n🔧 Step 2: Processing frames...")
    clothing_frames = []

    if args.debug:
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)

    for i, (base_frame, clothed_frame) in enumerate(zip(base_frames, clothed_frames)):
        # Align clothed frame to base
        aligned_clothed = align_frames(base_frame, clothed_frame)

        # Extract clothing
        clothing_frame = extract_clothing_layer(base_frame, aligned_clothed, args.tolerance)
        clothing_frames.append(clothing_frame)

        # Save debug frames if requested
        if args.debug:
            base_frame.save(debug_dir / f"frame_{i:02d}_base.png")
            clothed_frame.save(debug_dir / f"frame_{i:02d}_clothed.png")
            aligned_clothed.save(debug_dir / f"frame_{i:02d}_aligned.png")
            clothing_frame.save(debug_dir / f"frame_{i:02d}_clothing.png")

        print(f"   Processed frame {i+1}/{len(base_frames)}")

    # Step 3: Reassemble
    print("\n🔧 Step 3: Reassembling spritesheet...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reassemble_spritesheet(clothing_frames, args.output)
    print(f"   Saved to {args.output}")

    if args.debug:
        print(f"\n🐛 Debug frames saved to {debug_dir}/")

    print("\n✅ Extraction complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
