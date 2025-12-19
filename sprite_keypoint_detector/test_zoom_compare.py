#!/usr/bin/env python3
"""Create zoomed comparison of inpainting methods focusing on problem areas."""

import cv2
import numpy as np
from pathlib import Path


def extract_region(img: np.ndarray, center: tuple, size: int = 80) -> np.ndarray:
    """Extract a square region centered at given point."""
    cx, cy = center
    half = size // 2
    h, w = img.shape[:2]

    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    return img[y1:y2, x1:x2].copy()


def add_label(img: np.ndarray, label: str, position: str = "top") -> np.ndarray:
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if position == "top":
        cv2.putText(result, label, (5, 20), font, 0.5, (255, 255, 255, 255), 1)
    else:
        cv2.putText(result, label, (5, result.shape[0] - 10), font, 0.5, (255, 255, 255, 255), 1)
    return result


def main():
    base_dir = Path(__file__).parent.parent / "training_data" / "skeleton_comparison"

    # Regions of interest (approximate centers of problem areas)
    # These are in 512x512 coordinates
    regions = {
        "left_armpit": (200, 230),
        "right_arm": (310, 240),
        "left_arm": (175, 260),
    }

    for frame_name in ["clothed_frame_00", "clothed_frame_01"]:
        print(f"\n=== {frame_name} ===")

        # Load the comparison image (has all 3 versions side by side)
        comparison_path = base_dir / f"{frame_name}_texture_borrow.png"
        if not comparison_path.exists():
            print(f"  Not found: {comparison_path}")
            continue

        full_img = cv2.imread(str(comparison_path), cv2.IMREAD_UNCHANGED)
        h, w = full_img.shape[:2]

        # Split into 3 parts
        third = w // 3
        no_warp = full_img[:, :third]
        texture_borrow = full_img[:, third:2*third]
        opencv = full_img[:, 2*third:]

        # For each region, extract and compare
        for region_name, (cx, cy) in regions.items():
            print(f"  Extracting {region_name}...")

            # Extract from each version (scale up 4x for visibility)
            size = 60
            scale = 4

            no_warp_region = extract_region(no_warp, (cx, cy), size)
            texture_region = extract_region(texture_borrow, (cx, cy), size)
            opencv_region = extract_region(opencv, (cx, cy), size)

            # Scale up
            no_warp_scaled = cv2.resize(no_warp_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            texture_scaled = cv2.resize(texture_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            opencv_scaled = cv2.resize(opencv_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            # Add labels
            no_warp_labeled = add_label(no_warp_scaled, "no_warp")
            texture_labeled = add_label(texture_scaled, "texture_borrow")
            opencv_labeled = add_label(opencv_scaled, "opencv")

            # Combine
            zoom_comparison = np.hstack([no_warp_labeled, texture_labeled, opencv_labeled])

            output_path = base_dir / f"{frame_name}_{region_name}_zoom.png"
            cv2.imwrite(str(output_path), zoom_comparison)
            print(f"  Saved: {output_path.name}")


if __name__ == "__main__":
    main()
