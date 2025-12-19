#!/usr/bin/env python3
"""Test how shift+inpaint results look after pixelization.

Pixelization: downscale to target resolution then upscale back with nearest neighbor.
This simulates the final pixel art look.
"""

import cv2
import numpy as np
from pathlib import Path


def pixelize(image: np.ndarray, pixel_size: int = 4) -> np.ndarray:
    """Pixelize an image by downscaling then upscaling with nearest neighbor.

    Args:
        image: Input RGBA image
        pixel_size: How many pixels become one "pixel art pixel"
                   e.g., 4 means 512->128->512

    Returns:
        Pixelized image at original resolution
    """
    h, w = image.shape[:2]
    small_h, small_w = h // pixel_size, w // pixel_size

    # Downscale with area averaging (good for shrinking)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Upscale with nearest neighbor (crisp pixels)
    pixelized = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelized


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    result = img.copy()
    cv2.putText(result, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 255), 1)
    return result


def main():
    base_dir = Path(__file__).parent.parent / "training_data"
    output_dir = base_dir / "skeleton_comparison"

    # Process both the inpaint-only and shift+inpaint results
    for frame_name in ["clothed_frame_00", "clothed_frame_01"]:
        print(f"\n=== Pixelizing {frame_name} ===")

        images = {}

        # Load comparison images (these are composites on base)
        inpaint_path = output_dir / f"{frame_name}_inpaint.png"
        shift_inpaint_path = output_dir / f"{frame_name}_shift_inpaint.png"

        if inpaint_path.exists():
            # These are side-by-side comparisons, extract the right half (the result)
            full = cv2.imread(str(inpaint_path), cv2.IMREAD_UNCHANGED)
            h, w = full.shape[:2]
            images["inpaint"] = full[:, w//2:, :]

        if shift_inpaint_path.exists():
            full = cv2.imread(str(shift_inpaint_path), cv2.IMREAD_UNCHANGED)
            h, w = full.shape[:2]
            images["shift_inpaint"] = full[:, w//2:, :]

        # Also load the no-warp baseline (left half of any comparison)
        if shift_inpaint_path.exists():
            full = cv2.imread(str(shift_inpaint_path), cv2.IMREAD_UNCHANGED)
            h, w = full.shape[:2]
            images["no_warp"] = full[:, :w//2, :]

        if not images:
            print(f"  No images found for {frame_name}")
            continue

        # Pixelize each
        results = []
        for name, img in images.items():
            # Remove label area at top before pixelizing
            pixelized = pixelize(img, pixel_size=4)
            labeled = add_label(pixelized, f"{name} (pixelized)")
            results.append(labeled)
            print(f"  Pixelized: {name}")

        # Create comparison
        comparison = np.hstack(results)

        output_path = output_dir / f"{frame_name}_pixelized.png"
        cv2.imwrite(str(output_path), comparison)
        print(f"  Saved: {output_path.name}")

        # Also save individual pixelized images for closer inspection
        for name, img in images.items():
            pixelized = pixelize(img, pixel_size=4)
            cv2.imwrite(str(output_dir / f"{frame_name}_{name}_pixelized.png"), pixelized)


if __name__ == "__main__":
    main()
