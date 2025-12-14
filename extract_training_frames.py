#!/usr/bin/env python3
"""Extract all 25 frames from base and clothed spritesheets for training."""

import sys
from pathlib import Path
from PIL import Image


def extract_all_frames(spritesheet_path: str, output_dir: Path, prefix: str):
    """Extract all 25 frames from a 5x5 spritesheet."""
    spritesheet = Image.open(spritesheet_path)

    frames_per_row = 5
    frame_size = 512

    for frame_num in range(25):
        row = frame_num // frames_per_row
        col = frame_num % frames_per_row

        left = col * frame_size
        top = row * frame_size
        right = left + frame_size
        bottom = top + frame_size

        frame = spritesheet.crop((left, top, right, bottom))
        frame.save(output_dir / f"{prefix}_frame_{frame_num:02d}.png")

    print(f"✓ Extracted 25 frames from {spritesheet_path}")


def main():
    # Create output directory
    training_dir = Path("training_data")
    training_dir.mkdir(exist_ok=True)

    frames_dir = training_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Extract base frames
    extract_all_frames("examples/input/base.png", frames_dir, "base")

    # Extract clothed frames
    extract_all_frames("examples/input/reference.png", frames_dir, "clothed")

    print(f"\n✓ All frames extracted to {frames_dir}/")
    print(f"  - 25 base frames (base_frame_00.png to base_frame_24.png)")
    print(f"  - 25 clothed frames (clothed_frame_00.png to clothed_frame_24.png)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
