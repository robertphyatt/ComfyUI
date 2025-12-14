#!/usr/bin/env python3
"""Analyze image properties of OpenPose successful vs failed frames."""

from pathlib import Path
from PIL import Image, ImageStat
import json
import numpy as np

def analyze_frame(frame_path: Path) -> dict:
    """Extract image properties for comparison."""
    img = Image.open(frame_path)

    # Get bounding box of non-transparent/non-black pixels
    if img.mode == 'RGBA':
        alpha = img.split()[3]
        bbox = alpha.getbbox()
        # Analyze RGB channels only (exclude alpha)
        rgb_img = img.convert('RGB')
    else:
        # Convert to grayscale and find non-black regions
        gray = img.convert('L')
        bbox = gray.getbbox()
        rgb_img = img.convert('RGB')

    # Calculate sprite dimensions and position
    if bbox:
        sprite_width = bbox[2] - bbox[0]
        sprite_height = bbox[3] - bbox[1]
        sprite_center_x = (bbox[0] + bbox[2]) / 2
        sprite_center_y = (bbox[1] + bbox[3]) / 2

        # Crop to sprite region for detailed analysis
        sprite_region = rgb_img.crop(bbox)

        # Calculate statistics for sprite region
        stats = ImageStat.Stat(sprite_region)
        mean_brightness = sum(stats.mean) / 3  # Average across RGB
        stddev = sum(stats.stddev) / 3  # Average std dev

        # Convert to numpy for more analysis
        sprite_array = np.array(sprite_region)
        min_val = int(sprite_array.min())
        max_val = int(sprite_array.max())
        contrast_range = int(max_val - min_val)

    else:
        sprite_width = sprite_height = 0
        sprite_center_x = sprite_center_y = 0
        mean_brightness = stddev = min_val = max_val = contrast_range = 0

    # Overall image statistics
    overall_stats = ImageStat.Stat(rgb_img)

    return {
        "path": str(frame_path),
        "size": img.size,
        "mode": img.mode,
        "format": img.format,
        "bbox": bbox,
        "sprite_width": sprite_width,
        "sprite_height": sprite_height,
        "sprite_center": (sprite_center_x, sprite_center_y),
        "canvas_center": (img.width / 2, img.height / 2),
        "offset_from_center": (
            sprite_center_x - img.width / 2,
            sprite_center_y - img.height / 2
        ),
        "sprite_mean_brightness": mean_brightness,
        "sprite_stddev": stddev,
        "sprite_min_value": min_val,
        "sprite_max_value": max_val,
        "sprite_contrast_range": contrast_range,
        "overall_mean_brightness": sum(overall_stats.mean) / 3
    }

if __name__ == "__main__":
    frames_dir = Path("training_data/frames")

    # Analyze working frame
    working = analyze_frame(frames_dir / "base_frame_04.png")

    # Analyze failing frames
    failing = [
        analyze_frame(frames_dir / "base_frame_00.png"),
        analyze_frame(frames_dir / "base_frame_12.png")
    ]

    print("=== WORKING FRAME (04) ===")
    print(json.dumps(working, indent=2))

    print("\n=== FAILING FRAME (00) ===")
    print(json.dumps(failing[0], indent=2))

    print("\n=== FAILING FRAME (12) ===")
    print(json.dumps(failing[1], indent=2))

    # Calculate differences
    print("\n=== DIFFERENCES ===")
    compare_keys = [
        "sprite_width", "sprite_height", "sprite_center", "offset_from_center",
        "sprite_mean_brightness", "sprite_stddev", "sprite_contrast_range",
        "sprite_min_value", "sprite_max_value"
    ]
    for key in compare_keys:
        print(f"{key}:")
        print(f"  Frame 04: {working[key]}")
        print(f"  Frame 00: {failing[0][key]}")
        print(f"  Frame 12: {failing[1][key]}")

    # Highlight significant differences
    print("\n=== KEY DIFFERENCES (Frame 04 vs 00/12) ===")
    for key in compare_keys:
        if isinstance(working[key], (int, float)):
            diff_00 = abs(working[key] - failing[0][key])
            diff_12 = abs(working[key] - failing[1][key])
            if diff_00 > 1 or diff_12 > 1:
                print(f"{key}: Frame 04 differs by {diff_00:.2f} from 00, {diff_12:.2f} from 12")
