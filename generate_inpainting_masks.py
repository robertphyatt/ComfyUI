#!/usr/bin/env python3
"""Generate inpainting masks from alpha channel of base frames."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple


def generate_mask_from_alpha(image: Image.Image) -> Image.Image:
    """Generate binary inpainting mask from alpha channel.

    Args:
        image: RGBA image

    Returns:
        Binary mask (L mode): white = generate here, black = keep original
    """
    # Extract alpha channel
    if image.mode != 'RGBA':
        raise ValueError(f"Image must be RGBA, got {image.mode}")

    alpha = np.array(image)[:, :, 3]

    # Create binary mask: alpha > 0 → white (255), else black (0)
    mask = (alpha > 0).astype(np.uint8) * 255

    return Image.fromarray(mask, mode='L')


def generate_masks_for_frames(frames_dir: Path, output_dir: Path,
                              frame_range: Tuple[int, int] = (0, 25)) -> None:
    """Generate inpainting masks for base frames.

    Args:
        frames_dir: Directory containing base_frame_XX.png files
        output_dir: Directory to save mask_XX.png files
        frame_range: (start, end) frame indices (end exclusive)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    start, end = frame_range

    for frame_idx in range(start, end):
        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"

        if not base_path.exists():
            print(f"Warning: {base_path} not found, skipping")
            continue

        # Load base frame
        base_img = Image.open(base_path).convert('RGBA')

        # Generate mask
        mask = generate_mask_from_alpha(base_img)

        # Save
        output_path = output_dir / f"mask_{frame_idx:02d}.png"
        mask.save(output_path)
        print(f"Generated {output_path}")


def main():
    """Generate inpainting masks for all base frames."""
    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/masks_inpainting")

    print("=" * 70)
    print("GENERATING INPAINTING MASKS FROM ALPHA CHANNEL")
    print("=" * 70)
    print()

    generate_masks_for_frames(frames_dir, output_dir)

    print()
    print("=" * 70)
    print("✓ All masks generated")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
