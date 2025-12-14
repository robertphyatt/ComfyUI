#!/usr/bin/env python3
"""Interactive mask generation - Claude Code will analyze each frame."""

import json
from pathlib import Path
from PIL import Image
import numpy as np


def save_mask_from_rle(runs, output_path: Path):
    """Convert run-length encoding to mask image."""
    # Decode run-length encoding
    mask = np.zeros(512 * 512, dtype=np.uint8)
    pos = 0
    for value, count in runs:
        mask[pos:pos+count] = value
        pos += count

    # Reshape to 512x512
    mask = mask.reshape(512, 512)

    # Save mask
    mask_img = Image.fromarray(mask * 255, 'L')
    mask_img.save(output_path)

    # Statistics
    clothing_pixels = np.sum(mask == 1)
    percent = 100 * clothing_pixels / (512 * 512)
    return clothing_pixels, percent


def process_frame(frame_num: int, rle_data: dict):
    """Process a single frame's RLE data and save mask."""
    masks_dir = Path("training_data/masks_initial")
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_path = masks_dir / f"mask_{frame_num:02d}.png"

    runs = rle_data["runs"]
    clothing_pixels, percent = save_mask_from_rle(runs, mask_path)

    print(f"Frame {frame_num:02d}: ✓ Saved ({clothing_pixels} pixels, {percent:.1f}%)")
    return mask_path


def main():
    print("Interactive mask generation")
    print("=" * 70)
    print()
    print("This script will help Claude Code generate masks interactively.")
    print("Claude Code will analyze each frame and provide RLE data.")
    print()
    print("Ready to begin!")


if __name__ == "__main__":
    main()
