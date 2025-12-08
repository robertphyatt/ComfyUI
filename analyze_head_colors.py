#!/usr/bin/env python3
"""Analyze actual pixel colors in the head region."""

from PIL import Image
import numpy as np

def main():
    """Analyze head colors."""
    # Load aligned clothed frame
    img = Image.open("debug_frames/frame_12_aligned.png")
    img_arr = np.array(img)

    # Head bounding box from AI: (220,50) to (300,150)
    x_min, y_min = 220, 50
    x_max, y_max = 300, 150

    print("Analyzing head region pixels...")
    print(f"Bounding box: ({x_min},{y_min}) to ({x_max},{y_max})")
    print()

    # Collect all non-transparent pixels in this region
    colors = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixel = img_arr[y, x]
            # Skip transparent
            if pixel[3] == 0:
                continue
            r, g, b = pixel[:3]
            colors.append((r, g, b))

    if not colors:
        print("No non-transparent pixels found!")
        return 1

    colors = np.array(colors)

    print(f"Found {len(colors)} non-transparent pixels")
    print()
    print("RGB Statistics:")
    print(f"  R: min={colors[:,0].min()}, max={colors[:,0].max()}, mean={colors[:,0].mean():.1f}")
    print(f"  G: min={colors[:,1].min()}, max={colors[:,1].max()}, mean={colors[:,1].mean():.1f}")
    print(f"  B: min={colors[:,2].min()}, max={colors[:,2].max()}, mean={colors[:,2].mean():.1f}")
    print()

    print("AI provided ranges:")
    print("  R: 120-180")
    print("  G: 115-175")
    print("  B: 110-170")
    print()

    # Count how many pixels fall within AI's ranges
    in_range = 0
    for r, g, b in colors:
        if (120 <= r <= 180 and 115 <= g <= 175 and 110 <= b <= 170):
            in_range += 1

    print(f"Pixels matching AI's ranges: {in_range}/{len(colors)} ({100*in_range/len(colors):.1f}%)")
    print()

    # Show unique colors
    unique_colors = np.unique(colors, axis=0)
    print(f"Unique colors: {len(unique_colors)}")
    print("First 20 unique colors:")
    for i, (r, g, b) in enumerate(unique_colors[:20]):
        print(f"  RGB({r:3d}, {g:3d}, {b:3d})")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
