# generate_initial_masks.py
"""Generate initial segmentation masks using color-based segmentation."""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def generate_mask_from_color_diff(base_img: np.ndarray, clothed_img: np.ndarray) -> np.ndarray:
    """Generate binary mask from color difference between base and clothed frames.

    Args:
        base_img: Base character image (H, W, 3) RGB uint8
        clothed_img: Clothed character image (H, W, 3) RGB uint8

    Returns:
        Binary mask (H, W) uint8, where 1=clothing, 0=not-clothing
    """
    # Compute absolute difference
    diff = cv2.absdiff(clothed_img, base_img)

    # Sum across RGB channels to get total change
    diff_magnitude = np.sum(diff, axis=2)

    # Threshold: pixels with significant change
    changed_pixels = diff_magnitude > 30

    # Analyze colors in changed regions
    mask = np.zeros(base_img.shape[:2], dtype=np.uint8)

    for y in range(base_img.shape[0]):
        for x in range(base_img.shape[1]):
            if not changed_pixels[y, x]:
                continue

            # Get pixel color in clothed image
            r, g, b = clothed_img[y, x]

            # Check if brown-ish (armor color range)
            # Brown: R > G > B, with R in range 80-140
            is_brown = (r > g and g > b and 80 <= r <= 140)

            # Check if gray-ish (base character head)
            # Gray: R ≈ G ≈ B
            color_variance = np.std([r, g, b])
            is_gray = color_variance < 20

            # Mark as clothing if brown and not gray
            if is_brown and not is_gray:
                mask[y, x] = 1

    return mask


def generate_all_masks(frames_dir: Path, output_dir: Path):
    """Generate initial masks for all 25 frame pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_num in range(25):
        base_path = frames_dir / f"base_frame_{frame_num:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_num:02d}.png"
        mask_path = output_dir / f"mask_{frame_num:02d}.png"

        if mask_path.exists():
            print(f"Frame {frame_num:02d}: Skipping (already exists)")
            continue

        # Load images
        base = np.array(Image.open(base_path).convert('RGB'))
        clothed = np.array(Image.open(clothed_path).convert('RGB'))

        # Generate mask
        mask = generate_mask_from_color_diff(base, clothed)

        # Save mask
        mask_img = Image.fromarray(mask * 255, 'L')
        mask_img.save(mask_path)

        # Statistics
        clothing_pixels = np.sum(mask == 1)
        percent = 100 * clothing_pixels / (512 * 512)
        print(f"Frame {frame_num:02d}: ✓ Generated ({clothing_pixels} pixels, {percent:.1f}%)")


if __name__ == "__main__":
    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/masks_initial")

    print("Generating initial masks using color-based segmentation...")
    print("=" * 70)
    generate_all_masks(frames_dir, output_dir)
    print("=" * 70)
    print(f"✓ Initial masks saved to {output_dir}/")
    print("\nNext: Use mask_correction_tool.py to fix any errors")
