"""Extract palette from a spritesheet image.

Usage:
    python -m sprite_keypoint_detector.extract_palette <spritesheet.png> <output_palette.png>
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from .color_correction import extract_palette, save_palette_image


def extract_palette_from_spritesheet(
    spritesheet_path: Path,
    n_colors: int = 16
) -> np.ndarray:
    """Extract palette from a spritesheet image.

    Args:
        spritesheet_path: Path to spritesheet image
        n_colors: Number of colors to extract (default 16)

    Returns:
        Palette array of shape (n_colors, 3) with BGR values
    """
    img = cv2.imread(str(spritesheet_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {spritesheet_path}")

    # Use existing extract_palette function (expects list of frames)
    # Treat entire spritesheet as single frame
    if len(img.shape) == 2:
        # Grayscale - convert to BGRA
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        # BGR - add alpha channel (fully opaque)
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)

    return extract_palette([img], n_colors)


def main():
    parser = argparse.ArgumentParser(
        description="Extract palette from a spritesheet image"
    )
    parser.add_argument("spritesheet", type=Path,
                       help="Path to spritesheet image")
    parser.add_argument("output", type=Path,
                       help="Path to save palette image")
    parser.add_argument("--colors", type=int, default=16,
                       help="Number of colors to extract (default 16)")

    args = parser.parse_args()

    if not args.spritesheet.exists():
        print(f"Error: Spritesheet not found: {args.spritesheet}")
        return 1

    print(f"Extracting {args.colors}-color palette from {args.spritesheet}...")
    palette = extract_palette_from_spritesheet(args.spritesheet, args.colors)

    print(f"Saving palette to {args.output}...")
    save_palette_image(palette, args.output)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
