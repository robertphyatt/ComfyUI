#!/usr/bin/env python3
"""Extend brown armor to cover gray feet in aligned clothed frames.

For each frame:
1. Find where brown armor ends (bottom-most armor pixels)
2. Find where gray feet are (in base frame)
3. Extend armor downward to cover those feet completely
"""

import numpy as np
from PIL import Image
from pathlib import Path


def extend_armor_to_cover_feet(clothed_frame: Image.Image, base_frame: Image.Image) -> Image.Image:
    """Extend brown armor downward to cover gray feet.

    Args:
        clothed_frame: Aligned clothed frame with armor
        base_frame: Base frame showing where feet are

    Returns:
        Clothed frame with armor extended to cover feet
    """
    clothed_arr = np.array(clothed_frame.convert('RGBA'))
    base_arr = np.array(base_frame.convert('RGBA'))

    height, width = clothed_arr.shape[:2]

    # For each column, extend armor downward
    for x in range(width):
        # Find bottom-most armor pixel in this column
        armor_alpha = clothed_arr[:, x, 3]
        armor_present = armor_alpha > 128

        if not armor_present.any():
            continue  # No armor in this column

        # Find last (bottom-most) armor pixel
        armor_bottom_y = np.max(np.where(armor_present)[0])

        # Find if there are gray feet below the armor in base frame
        base_alpha = base_arr[:, x, 3]
        base_present = base_alpha > 128

        if not base_present.any():
            continue  # No base character in this column

        # Find bottom-most base pixel (where feet end)
        base_bottom_y = np.max(np.where(base_present)[0])

        # If base extends below armor, we need to extend armor
        if base_bottom_y > armor_bottom_y:
            # Get the armor pixel color at the bottom
            armor_color = clothed_arr[armor_bottom_y, x].copy()

            # Extend armor downward from armor_bottom_y to base_bottom_y
            for y in range(armor_bottom_y + 1, base_bottom_y + 1):
                if y >= height:
                    break

                # Use the bottom armor pixel color
                clothed_arr[y, x] = armor_color

    return Image.fromarray(clothed_arr)


def main():
    """Extend armor to cover feet in all aligned frames."""
    frames_dir = Path("training_data/frames")
    aligned_dir = Path("training_data/frames_aligned")
    output_dir = Path("training_data/frames_complete")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXTENDING ARMOR TO COVER FEET")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        # Load frames
        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        aligned_path = aligned_dir / f"clothed_frame_{frame_idx:02d}.png"

        base = Image.open(base_path)
        aligned = Image.open(aligned_path)

        # Extend armor
        extended = extend_armor_to_cover_feet(aligned, base)

        # Count pixels before and after
        aligned_arr = np.array(aligned)
        extended_arr = np.array(extended)

        before_pixels = np.sum(aligned_arr[:, :, 3] > 128)
        after_pixels = np.sum(extended_arr[:, :, 3] > 128)
        added_pixels = after_pixels - before_pixels

        print(f"  Before: {before_pixels:6d} pixels")
        print(f"  After:  {after_pixels:6d} pixels")
        print(f"  Added:  {added_pixels:6d} pixels (+{100*added_pixels/before_pixels:.1f}%)")

        # Save
        output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
        extended.save(output_path)
        print(f"  ✓ Saved to {output_path}")
        print()

    print("=" * 70)
    print(f"✓ All frames with extended armor saved to {output_dir}/")
    print("=" * 70)
    print()
    print("Next: Extract clothing from these complete frames")


if __name__ == "__main__":
    main()
