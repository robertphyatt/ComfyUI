#!/usr/bin/env python3
"""Align clothed frames to base frames - FIX THE FUNDAMENTAL MISTAKE.

For each frame:
1. Find character bounding box in base frame
2. Find character bounding box in clothed frame
3. Calculate offset to align them
4. Shift clothed frame so character positions match EXACTLY
"""

import numpy as np
from PIL import Image
from pathlib import Path


def find_character_bounds(frame_rgba: np.ndarray, alpha_threshold: int = 128) -> tuple[int, int, int, int] | None:
    """Find bounding box of non-transparent pixels.

    Returns:
        (left, top, right, bottom) or None if no character found
    """
    alpha = frame_rgba[:, :, 3]

    # Find non-transparent pixels
    mask = alpha > alpha_threshold

    if not mask.any():
        return None

    # Find bounds
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    return (left, top, right, bottom)


def align_clothed_to_base(base_frame: Image.Image, clothed_frame: Image.Image) -> Image.Image:
    """Align clothed frame to match base frame character position EXACTLY.

    Args:
        base_frame: Base character frame (position reference)
        clothed_frame: Clothed character frame (needs alignment)

    Returns:
        Aligned clothed frame where character position matches base
    """
    # Convert to numpy
    base_arr = np.array(base_frame.convert('RGBA'))
    clothed_arr = np.array(clothed_frame.convert('RGBA'))

    # Find character bounds in both
    base_bounds = find_character_bounds(base_arr)
    clothed_bounds = find_character_bounds(clothed_arr)

    if base_bounds is None or clothed_bounds is None:
        print(f"  WARNING: Could not find character in one of the frames")
        return clothed_frame

    base_left, base_top, base_right, base_bottom = base_bounds
    cloth_left, cloth_top, cloth_right, cloth_bottom = clothed_bounds

    # Calculate character center in each frame
    base_center_x = (base_left + base_right) / 2
    base_center_y = (base_top + base_bottom) / 2

    cloth_center_x = (cloth_left + cloth_right) / 2
    cloth_center_y = (cloth_top + cloth_bottom) / 2

    # Calculate offset needed to align centers
    offset_x = int(base_center_x - cloth_center_x)
    offset_y = int(base_center_y - cloth_center_y)

    print(f"  Base center: ({base_center_x:.1f}, {base_center_y:.1f})")
    print(f"  Clothed center: ({cloth_center_x:.1f}, {cloth_center_y:.1f})")
    print(f"  Offset needed: ({offset_x:+d}, {offset_y:+d})")

    # Create aligned frame by shifting clothed frame
    height, width = clothed_arr.shape[:2]
    aligned = np.zeros_like(clothed_arr)

    # Calculate source and destination regions for the shift
    if offset_x >= 0 and offset_y >= 0:
        # Shift right and down
        dst_slice_y = slice(offset_y, height)
        dst_slice_x = slice(offset_x, width)
        src_slice_y = slice(0, height - offset_y)
        src_slice_x = slice(0, width - offset_x)
    elif offset_x >= 0 and offset_y < 0:
        # Shift right and up
        dst_slice_y = slice(0, height + offset_y)
        dst_slice_x = slice(offset_x, width)
        src_slice_y = slice(-offset_y, height)
        src_slice_x = slice(0, width - offset_x)
    elif offset_x < 0 and offset_y >= 0:
        # Shift left and down
        dst_slice_y = slice(offset_y, height)
        dst_slice_x = slice(0, width + offset_x)
        src_slice_y = slice(0, height - offset_y)
        src_slice_x = slice(-offset_x, width)
    else:
        # Shift left and up
        dst_slice_y = slice(0, height + offset_y)
        dst_slice_x = slice(0, width + offset_x)
        src_slice_y = slice(-offset_y, height)
        src_slice_x = slice(-offset_x, width)

    # Apply shift
    aligned[dst_slice_y, dst_slice_x] = clothed_arr[src_slice_y, src_slice_x]

    return Image.fromarray(aligned)


def main():
    """Align all 25 clothed frames to match base frame positions."""
    frames_dir = Path("training_data/frames")
    aligned_dir = Path("training_data/frames_aligned")
    aligned_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ALIGNING CLOTHED FRAMES TO BASE FRAMES")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        # Load frames
        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_idx:02d}.png"

        if not base_path.exists():
            print(f"  ERROR: Base frame not found: {base_path}")
            continue

        if not clothed_path.exists():
            print(f"  ERROR: Clothed frame not found: {clothed_path}")
            continue

        base = Image.open(base_path)
        clothed = Image.open(clothed_path)

        # Align clothed to base
        aligned = align_clothed_to_base(base, clothed)

        # Save aligned frame
        output_path = aligned_dir / f"clothed_frame_{frame_idx:02d}.png"
        aligned.save(output_path)
        print(f"  ✓ Saved to {output_path}")
        print()

    print("=" * 70)
    print(f"✓ All 25 frames aligned and saved to {aligned_dir}/")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Visually verify alignment by overlaying frames")
    print("2. Extend armor to cover feet in aligned frames")
    print("3. Re-extract masks from aligned frames")


if __name__ == "__main__":
    main()
