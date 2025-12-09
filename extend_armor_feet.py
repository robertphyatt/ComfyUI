#!/usr/bin/env python3
"""Extend armor to cover feet in OpenPose-aligned frames."""

import numpy as np
from PIL import Image
from pathlib import Path


def extend_armor_to_cover_feet(clothed_frame: Image.Image, base_frame: Image.Image) -> Image.Image:
    """Extend brown armor downward to cover gray feet.

    For each column, extend armor from its bottom-most pixel down to cover
    any base character pixels below it.
    """
    clothed_arr = np.array(clothed_frame.convert('RGBA'))
    base_arr = np.array(base_frame.convert('RGBA'))

    height, width = clothed_arr.shape[:2]

    # For each column, extend armor downward
    for x in range(width):
        # Find bottom-most armor pixel
        armor_alpha = clothed_arr[:, x, 3]
        armor_present = armor_alpha > 128

        if not armor_present.any():
            continue

        armor_bottom_y = np.max(np.where(armor_present)[0])

        # Find bottom-most base pixel (where feet end)
        base_alpha = base_arr[:, x, 3]
        base_present = base_alpha > 128

        if not base_present.any():
            continue

        base_bottom_y = np.max(np.where(base_present)[0])

        # Extend armor if base extends below it
        if base_bottom_y > armor_bottom_y:
            armor_color = clothed_arr[armor_bottom_y, x].copy()

            for y in range(armor_bottom_y + 1, min(base_bottom_y + 1, height)):
                clothed_arr[y, x] = armor_color

    return Image.fromarray(clothed_arr)


def main():
    """Extend armor in all IPAdapter-generated frames."""
    frames_dir = Path("training_data/frames")
    generated_dir = Path("training_data/frames_ipadapter_generated")
    output_dir = Path("training_data/frames_complete_ipadapter")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXTENDING ARMOR TO COVER FEET")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        generated_path = generated_dir / f"clothed_frame_{frame_idx:02d}.png"

        base = Image.open(base_path)
        generated = Image.open(generated_path)

        # Extend armor
        extended = extend_armor_to_cover_feet(generated, base)

        # Save
        output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
        extended.save(output_path)

        # Count pixels
        extended_arr = np.array(extended)
        pixels = np.sum(extended_arr[:, :, 3] > 128)
        print(f"  {pixels:6d} pixels")
        print(f"  ✓ Saved to {output_path}")
        print()

    print("=" * 70)
    print("✓ Armor extended for all frames")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()
