#!/usr/bin/env python3
"""Validate and correct model-predicted masks using the interactive tool."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mask_correction_tool import MaskEditor
import matplotlib.pyplot as plt


def main():
    """Launch mask correction tool for predicted masks."""
    frames_dir = Path("training_data_validation/frames")
    masks_dir = Path("training_data_validation/masks_corrected")

    print("=" * 70)
    print("VALIDATING MODEL-PREDICTED MASKS")
    print("=" * 70)
    print()
    print("Controls:")
    print("  Left Click: Add clothing pixels (paint red)")
    print("  Right Click: Remove clothing pixels (erase)")
    print("  Ctrl + Scroll: Zoom in/out")
    print("  +/- Keys: Zoom in/out")
    print("  0 Key: Reset zoom")
    print("  Cmd+Z / Ctrl+Z: Undo")
    print("  Cmd+Shift+Z / Ctrl+Shift+Z: Redo")
    print("  Save Button: Save corrections and move to next frame")
    print("  Cancel Button: Skip without saving")
    print()
    print("=" * 70)
    print()

    # Process all 25 frames
    for frame_idx in range(25):
        base_path = Path("training_data/frames") / f"base_frame_{frame_idx:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_idx:02d}.png"
        mask_path = masks_dir / f"mask_{frame_idx:02d}.png"

        if not base_path.exists():
            print(f"Skipping frame {frame_idx:02d}: base frame not found")
            continue

        if not clothed_path.exists():
            print(f"Skipping frame {frame_idx:02d}: clothed frame not found")
            continue

        if not mask_path.exists():
            print(f"Skipping frame {frame_idx:02d}: predicted mask not found")
            continue

        print(f"Reviewing frame {frame_idx:02d}...")

        # Load images as numpy arrays
        base_img = np.array(Image.open(base_path).convert('RGB'))
        clothed_img = np.array(Image.open(clothed_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Convert mask to binary (0 or 1)
        mask = (mask > 128).astype(np.uint8)

        # Launch correction tool
        tool = MaskEditor(
            base_img=base_img,
            clothed_img=clothed_img,
            mask=mask
        )

        plt.show()

        # Save corrected mask
        corrected_mask = (tool.mask * 255).astype(np.uint8)
        Image.fromarray(corrected_mask).save(mask_path)
        print(f"  → Saved to {mask_path}")

    print()
    print("=" * 70)
    print("✓ Mask validation complete!")
    print("=" * 70)
    print()
    print(f"Corrected masks saved in: {masks_dir}/")

    return 0


if __name__ == "__main__":
    main()
