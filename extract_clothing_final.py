#!/usr/bin/env python3
"""Extract final clothing spritesheet from OpenPose-aligned frames."""

import numpy as np
from PIL import Image
from pathlib import Path


def extract_with_validated_masks(frames_dir: Path, masks_dir: Path, output_path: Path):
    """Extract clothing using validated masks."""
    frame_size = 512
    spritesheet = Image.new('RGBA', (frame_size * 5, frame_size * 5), (0, 0, 0, 0))

    for frame_idx in range(25):
        clothed = np.array(Image.open(frames_dir / f'clothed_frame_{frame_idx:02d}.png').convert('RGBA'))
        mask = np.array(Image.open(masks_dir / f'mask_{frame_idx:02d}.png').convert('L'))

        # Apply mask
        clothing = clothed.copy()
        clothing[:, :, 3] = np.where(mask > 128, clothed[:, :, 3], 0)

        # Paste into spritesheet
        clothing_img = Image.fromarray(clothing)
        row = frame_idx // 5
        col = frame_idx % 5
        spritesheet.paste(clothing_img, (col * frame_size, row * frame_size), clothing_img)

    spritesheet.save(output_path)
    print(f"✓ Extracted clothing spritesheet to {output_path}")


def main():
    """Extract clothing spritesheet from complete OpenPose-aligned frames."""
    complete_dir = Path("training_data/frames_complete_openpose")
    output_path = Path("training_data/clothing_spritesheet_openpose.png")

    # Create 5x5 spritesheet
    frame_size = 512
    spritesheet = Image.new('RGBA', (frame_size * 5, frame_size * 5), (0, 0, 0, 0))

    print("=" * 70)
    print("EXTRACTING FINAL CLOTHING SPRITESHEET (OpenPose-aligned)")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        # Load complete frame
        complete = np.array(Image.open(complete_dir / f"clothed_frame_{frame_idx:02d}.png").convert('RGBA'))

        # Convert to image
        clothing_img = Image.fromarray(complete)

        # Calculate position in spritesheet
        row = frame_idx // 5
        col = frame_idx % 5
        x = col * frame_size
        y = row * frame_size

        # Paste into spritesheet
        spritesheet.paste(clothing_img, (x, y), clothing_img)

        pixels = np.sum(complete[:, :, 3] > 128)
        print(f"Frame {frame_idx:02d}: {pixels:6d} clothing pixels")

    # Save
    spritesheet.save(output_path)

    print()
    print("=" * 70)
    print(f"✓ Final clothing spritesheet saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
