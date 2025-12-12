#!/usr/bin/env python3
"""Deep pixel analysis to find what makes frame 4 detectable."""

from pathlib import Path
from PIL import Image
import numpy as np
import json

def deep_analyze(frame_path: Path) -> dict:
    """Perform deep pixel-level analysis."""
    img = Image.open(frame_path).convert('RGB')
    arr = np.array(img)

    # Find the actual sprite pixels (non-black in RGB)
    sprite_mask = np.any(arr > 10, axis=2)
    sprite_pixels = arr[sprite_mask]

    # Analyze color distribution
    if len(sprite_pixels) > 0:
        r_mean, g_mean, b_mean = sprite_pixels.mean(axis=0)
        r_std, g_std, b_std = sprite_pixels.std(axis=0)

        # Check for skin tone pixels (typically higher R, moderate G, lower B)
        # Skin tone approximation: R > 95, G > 40, B > 20, R > G, R > B
        skin_like = (sprite_pixels[:, 0] > 95) & \
                    (sprite_pixels[:, 1] > 40) & \
                    (sprite_pixels[:, 2] > 20) & \
                    (sprite_pixels[:, 0] > sprite_pixels[:, 1]) & \
                    (sprite_pixels[:, 0] > sprite_pixels[:, 2])
        skin_pixel_count = int(np.sum(skin_like))
        skin_percentage = (skin_pixel_count / len(sprite_pixels)) * 100

        # Get actual pixel value histogram
        unique_colors = len(np.unique(sprite_pixels.reshape(-1, 3), axis=0))

    else:
        r_mean = g_mean = b_mean = 0
        r_std = g_std = b_std = 0
        skin_pixel_count = 0
        skin_percentage = 0
        unique_colors = 0

    return {
        "path": str(frame_path),
        "total_sprite_pixels": int(np.sum(sprite_mask)),
        "rgb_means": [float(r_mean), float(g_mean), float(b_mean)],
        "rgb_stds": [float(r_std), float(g_std), float(b_std)],
        "unique_colors": unique_colors,
        "skin_tone_pixels": skin_pixel_count,
        "skin_tone_percentage": skin_percentage
    }

if __name__ == "__main__":
    frames_dir = Path("training_data/frames")

    frames = {
        0: frames_dir / "base_frame_00.png",
        4: frames_dir / "base_frame_04.png",
        12: frames_dir / "base_frame_12.png"
    }

    results = {}
    for frame_num, frame_path in frames.items():
        print(f"=== FRAME {frame_num:02d} ===")
        results[frame_num] = deep_analyze(frame_path)
        print(json.dumps(results[frame_num], indent=2))
        print()

    # Highlight key differences
    print("=== KEY DIFFERENCES ===")
    print(f"Skin tone pixels:")
    for frame_num in sorted(results.keys()):
        r = results[frame_num]
        print(f"  Frame {frame_num:02d}: {r['skin_tone_pixels']:6d} pixels ({r['skin_tone_percentage']:.2f}%)")

    print(f"\nUnique colors:")
    for frame_num in sorted(results.keys()):
        r = results[frame_num]
        print(f"  Frame {frame_num:02d}: {r['unique_colors']:6d} colors")

    print(f"\nRGB means (R, G, B):")
    for frame_num in sorted(results.keys()):
        r = results[frame_num]
        print(f"  Frame {frame_num:02d}: ({r['rgb_means'][0]:.2f}, {r['rgb_means'][1]:.2f}, {r['rgb_means'][2]:.2f})")
