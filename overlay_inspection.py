#!/usr/bin/env python3
"""Overlay clothing spritesheet on base to inspect for base character bleed-through."""

from PIL import Image
import numpy as np

# Load both spritesheets
base = Image.open('/Users/roberthyatt/Code/ComfyUI/examples/input/base.png').convert('RGBA')
clothing = Image.open('/Users/roberthyatt/Code/ComfyUI/training_data/clothing_spritesheet.png').convert('RGBA')

# Ensure same size
if base.size != clothing.size:
    print(f"Warning: Size mismatch - base: {base.size}, clothing: {clothing.size}")
    # Resize clothing to match base if needed
    if clothing.size != base.size:
        clothing = clothing.resize(base.size, Image.Resampling.NEAREST)

# Create composite: base + clothing overlay
composite = Image.alpha_composite(base, clothing)

# Save composite
output_path = '/Users/roberthyatt/Code/ComfyUI/training_data/overlay_inspection.png'
composite.save(output_path)
print(f"Overlay saved to: {output_path}")

# Also create a difference image to highlight where clothing pixels exist
clothing_np = np.array(clothing)
alpha = clothing_np[:, :, 3]

# Count non-transparent pixels per frame
width, height = clothing.size
frame_width = width // 5
frame_height = height // 5

print("\nClothing pixels per frame:")
for row in range(5):
    for col in range(5):
        frame_idx = row * 5 + col
        x1 = col * frame_width
        y1 = row * frame_height
        x2 = x1 + frame_width
        y2 = y1 + frame_height

        frame_alpha = alpha[y1:y2, x1:x2]
        pixel_count = np.sum(frame_alpha > 128)
        print(f"Frame {frame_idx:02d}: {pixel_count:6d} pixels", end="  ")
        if (frame_idx + 1) % 5 == 0:
            print()
