# generate_initial_masks.py
"""Generate initial segmentation masks using color-based segmentation."""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Color detection thresholds
DIFF_THRESHOLD = 30  # RGB difference magnitude threshold for changed pixels
BROWN_R_MIN = 80  # Minimum red channel value for brown armor
BROWN_R_MAX = 140  # Maximum red channel value for brown armor
COLOR_RANGE_GRAY_THRESHOLD = 30  # Max color range for gray detection (approximates std < 20)


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
    changed_pixels = diff_magnitude > DIFF_THRESHOLD

    # Extract RGB channels for vectorized operations
    r = clothed_img[:, :, 0]
    g = clothed_img[:, :, 1]
    b = clothed_img[:, :, 2]

    # Check brown condition (vectorized)
    # Brown: R > G > B, with R in range BROWN_R_MIN-BROWN_R_MAX
    is_brown = (r > g) & (g > b) & (r >= BROWN_R_MIN) & (r <= BROWN_R_MAX)

    # Check gray condition using max-min as proxy for std (much faster)
    # Gray: R ≈ G ≈ B (low color variance)
    color_range = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    is_gray = color_range < COLOR_RANGE_GRAY_THRESHOLD

    # Combine conditions: changed pixels that are brown and not gray
    mask = np.zeros(base_img.shape[:2], dtype=np.uint8)
    mask[changed_pixels & is_brown & ~is_gray] = 1

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

        try:
            # Load images
            base = np.array(Image.open(base_path).convert('RGB'))
            clothed = np.array(Image.open(clothed_path).convert('RGB'))
        except (FileNotFoundError, OSError) as e:
            print(f"Frame {frame_num:02d}: ✗ Error loading images: {e}")
            continue

        # Generate mask
        mask = generate_mask_from_color_diff(base, clothed)

        # Save mask
        mask_img = Image.fromarray(mask * 255)
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
