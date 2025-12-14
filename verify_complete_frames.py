#!/usr/bin/env python3
"""Verify complete frames show proper alignment AND foot coverage."""

import numpy as np
from PIL import Image
from pathlib import Path


def create_complete_verification(frame_idx: int, output_dir: Path):
    """Create verification: base | complete_clothed | overlay.

    This shows:
    - Alignment is correct
    - Armor covers feet (no gray bleeding through)
    """
    frames_dir = Path("training_data/frames")
    complete_dir = Path("training_data/frames_complete")

    # Load frames
    base = Image.open(frames_dir / f"base_frame_{frame_idx:02d}.png").convert('RGBA')
    complete = Image.open(complete_dir / f"clothed_frame_{frame_idx:02d}.png").convert('RGBA')

    # Create overlay
    overlay = Image.alpha_composite(base, complete)

    # Create side-by-side
    width, height = base.size
    comparison = Image.new('RGBA', (width * 3, height), (255, 255, 255, 255))

    comparison.paste(base, (0, 0))
    comparison.paste(complete, (width, 0))
    comparison.paste(overlay, (width * 2, 0))

    # Save
    output_path = output_dir / f"complete_frame_{frame_idx:02d}.png"
    comparison.save(output_path)
    return output_path


def main():
    """Generate verification images for complete frames."""
    output_dir = Path("training_data/complete_verification")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("VERIFYING COMPLETE FRAMES (Aligned + Armor Extended)")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        path = create_complete_verification(frame_idx, output_dir)
        print(f"Frame {frame_idx:02d}: ✓ {path}")

    print()
    print("=" * 70)
    print(f"✓ Complete verification images saved to {output_dir}/")
    print("=" * 70)
    print()
    print("Each image shows: Base | Complete Clothed | Overlay")
    print("Verify:")
    print("  1. Character positions align correctly")
    print("  2. Armor covers feet completely (no gray bleed-through)")


if __name__ == "__main__":
    main()
