#!/usr/bin/env python3
"""Test RemBG cloth segmentation on a single frame."""

import time
from pathlib import Path
from PIL import Image
import numpy as np
from rembg import new_session, remove


def extract_frame_from_spritesheet(spritesheet: Image.Image, frame_num: int,
                                   frames_per_row: int = 5, frame_size: int = 512) -> Image.Image:
    """Extract a single frame from a spritesheet."""
    row = frame_num // frames_per_row
    col = frame_num % frames_per_row

    left = col * frame_size
    top = row * frame_size
    right = left + frame_size
    bottom = top + frame_size

    return spritesheet.crop((left, top, right, bottom))


def test_rembg_cloth_segmentation():
    """Test RemBG with u2net_cloth_seg model."""

    print("Testing RemBG cloth segmentation...")
    print()

    # Load sprite sheets
    print("Loading sprite sheets...")
    base_sheet = Image.open("examples/input/base.png")
    clothed_sheet = Image.open("examples/input/reference.png")

    # Extract frame 0
    print("Extracting frame 0...")
    base_frame = extract_frame_from_spritesheet(base_sheet, 0)
    clothed_frame = extract_frame_from_spritesheet(clothed_sheet, 0)

    print(f"  Base frame size: {base_frame.size}")
    print(f"  Clothed frame size: {clothed_frame.size}")
    print(f"  Clothed frame mode: {clothed_frame.mode}")

    # Ensure RGB mode
    if clothed_frame.mode != 'RGB':
        clothed_frame = clothed_frame.convert('RGB')
        print(f"  Converted to RGB")

    # Test cloth segmentation
    print("Running RemBG u2net_cloth_seg model...")
    start_time = time.time()

    session = new_session('u2net_cloth_seg')

    # Try with alpha_matting disabled and explicit size
    result = remove(
        clothed_frame,
        session=session,
        alpha_matting=False,
        only_mask=False
    )

    print(f"  Input size: {clothed_frame.size}")
    print(f"  Result size: {result.size}")
    print(f"  Result mode: {result.mode}")

    # If result size doesn't match, crop/resize it
    if result.size != clothed_frame.size:
        print(f"  WARNING: Result size mismatch! Cropping to match input...")
        # Crop from center if larger
        if result.size[0] >= clothed_frame.size[0] and result.size[1] >= clothed_frame.size[1]:
            left = (result.size[0] - clothed_frame.size[0]) // 2
            top = (result.size[1] - clothed_frame.size[1]) // 2
            right = left + clothed_frame.size[0]
            bottom = top + clothed_frame.size[1]
            result = result.crop((left, top, right, bottom))
            print(f"  Cropped to: {result.size}")

    elapsed = time.time() - start_time
    print(f"✓ Segmentation completed in {elapsed:.2f}s")

    # Extract alpha channel as mask
    result_arr = np.array(result)
    alpha_channel = result_arr[:, :, 3]

    # Convert to binary mask
    mask = (alpha_channel > 128).astype(np.uint8)

    print()
    print("Results:")
    print(f"  Output shape: {result.size}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Clothing pixels (1): {(mask == 1).sum()}")
    print(f"  Base pixels (0): {(mask == 0).sum()}")
    print(f"  Total pixels: {mask.size}")

    # Save outputs for visual inspection
    output_dir = Path("debug_rembg_test")
    output_dir.mkdir(exist_ok=True)

    base_frame.save(output_dir / "frame_00_base.png")
    clothed_frame.save(output_dir / "frame_00_clothed.png")
    result.save(output_dir / "frame_00_rembg_result.png")

    # Create visualization of mask
    mask_vis = Image.fromarray(mask * 255, mode='L')
    mask_vis.save(output_dir / "frame_00_mask.png")

    # Create clothing-only output by applying mask
    clothed_arr = np.array(clothed_frame.convert('RGBA'))
    clothing_arr = clothed_arr.copy()

    print(f"  Clothed array shape: {clothed_arr.shape}")
    print(f"  Mask shape: {mask.shape}")

    # Ensure mask matches image dimensions
    if mask.shape != clothed_arr.shape[:2]:
        print(f"  WARNING: Mask shape {mask.shape} doesn't match image {clothed_arr.shape[:2]}")
        # Use result's alpha channel directly
        clothing_arr[:, :, 3] = result_arr[:, :, 3]
    else:
        clothing_arr[:, :, 3] = mask * 255  # Apply mask to alpha channel

    clothing_only = Image.fromarray(clothing_arr, 'RGBA')
    clothing_only.save(output_dir / "frame_00_clothing_only.png")

    print()
    print("Outputs saved to debug_rembg_test/:")
    print("  frame_00_base.png - Base character frame")
    print("  frame_00_clothed.png - Clothed character frame")
    print("  frame_00_rembg_result.png - RemBG output (RGBA)")
    print("  frame_00_mask.png - Binary mask visualization")
    print("  frame_00_clothing_only.png - Clothing extracted with mask")
    print()
    print("To view results:")
    print("  open debug_rembg_test/")


if __name__ == "__main__":
    test_rembg_cloth_segmentation()
