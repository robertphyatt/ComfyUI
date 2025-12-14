#!/usr/bin/env python3
"""Test difference-based segmentation approaches."""

import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
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


def test_option1_diff_to_rembg():
    """Option 1: Feed difference image to RemBG."""
    print("=" * 70)
    print("OPTION 1: Feed Difference Image to RemBG")
    print("=" * 70)
    print()

    # Load frames
    print("Loading frames...")
    base_sheet = Image.open("examples/input/base.png")
    clothed_sheet = Image.open("examples/input/reference.png")

    base_frame = extract_frame_from_spritesheet(base_sheet, 0)
    clothed_frame = extract_frame_from_spritesheet(clothed_sheet, 0)

    # Compute difference
    print("Computing difference image...")
    base_arr = np.array(base_frame.convert('RGB'))
    clothed_arr = np.array(clothed_frame.convert('RGB'))

    diff_arr = cv2.absdiff(clothed_arr, base_arr)
    diff_image = Image.fromarray(diff_arr, 'RGB')

    # Show difference stats
    diff_magnitude = np.sum(diff_arr, axis=2)
    print(f"  Difference pixels (any change): {np.sum(diff_magnitude > 0)}")
    print(f"  Difference pixels (significant): {np.sum(diff_magnitude > 30)}")

    # Try RemBG on difference image with different models
    output_dir = Path("debug_diff_test")
    output_dir.mkdir(exist_ok=True)

    # Save inputs
    base_frame.save(output_dir / "frame_00_base.png")
    clothed_frame.save(output_dir / "frame_00_clothed.png")
    diff_image.save(output_dir / "frame_00_diff.png")

    models_to_try = ['u2net', 'u2net_cloth_seg', 'isnet-general-use']

    for model_name in models_to_try:
        print(f"\nTrying RemBG with {model_name} on difference image...")
        start_time = time.time()

        try:
            session = new_session(model_name)
            result = remove(diff_image, session=session, alpha_matting=False)
            elapsed = time.time() - start_time

            print(f"  ✓ Completed in {elapsed:.2f}s")

            # Crop if needed
            if result.size != (512, 512):
                left = (result.size[0] - 512) // 2
                top = (result.size[1] - 512) // 2
                result = result.crop((left, top, left + 512, top + 512))

            # Extract mask
            result_arr = np.array(result)
            mask = (result_arr[:, :, 3] > 128).astype(np.uint8)

            print(f"  Mask stats:")
            print(f"    Clothing pixels: {np.sum(mask == 1)}")
            print(f"    Base pixels: {np.sum(mask == 0)}")
            print(f"    Percentage clothing: {100 * np.sum(mask == 1) / mask.size:.1f}%")

            # Save outputs
            result.save(output_dir / f"option1_{model_name}_result.png")
            mask_vis = Image.fromarray(mask * 255, 'L')
            mask_vis.save(output_dir / f"option1_{model_name}_mask.png")

            # Apply mask to clothed frame
            clothing_arr = np.array(clothed_frame.convert('RGBA'))
            clothing_arr[:, :, 3] = mask * 255
            clothing_only = Image.fromarray(clothing_arr, 'RGBA')
            clothing_only.save(output_dir / f"option1_{model_name}_clothing.png")

        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_option2_sam_automatic():
    """Option 2: SAM with automatic mask generation."""
    print()
    print("=" * 70)
    print("OPTION 2: SAM Automatic Mask Generation")
    print("=" * 70)
    print()

    try:
        import sys
        import torch
        sys.path.insert(0, "custom_nodes/comfyui_controlnet_aux/src")
        from custom_controlnet_aux.sam import SamDetector

        # Determine device (use CPU for SAM - MPS has dtype issues)
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU forced due to MPS float64 limitation)")
    except ImportError as e:
        print(f"✗ Cannot import SAM: {e}")
        print("  SAM might not be installed or available")
        import traceback
        traceback.print_exc()
        return

    # Load frames
    print("Loading frames...")
    base_sheet = Image.open("examples/input/base.png")
    clothed_sheet = Image.open("examples/input/reference.png")

    base_frame = extract_frame_from_spritesheet(base_sheet, 0)
    clothed_frame = extract_frame_from_spritesheet(clothed_sheet, 0)

    output_dir = Path("debug_diff_test")
    output_dir.mkdir(exist_ok=True)

    # Try SAM on clothed frame
    print("Loading SAM model...")
    start_time = time.time()

    try:
        sam = SamDetector.from_pretrained().to(device)
        print(f"  ✓ Model loaded in {time.time() - start_time:.2f}s")

        print("\nRunning SAM on clothed frame...")
        start_time = time.time()

        # Convert to format SAM expects
        clothed_np = np.array(clothed_frame)

        # Generate automatic masks
        masks = sam.generate_automatic_masks(clothed_np)
        print(f"  ✓ Generated {len(masks)} masks in {time.time() - start_time:.2f}s")

        # Sort by area (largest first)
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Save top 10 largest masks
        print(f"\nTop masks by area:")
        for i, mask_data in enumerate(masks_sorted[:10]):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            mask_img = Image.fromarray(mask, 'L')
            mask_img.save(output_dir / f"option2_sam_mask_{i:02d}.png")

            area_pct = 100 * mask_data['area'] / (512 * 512)
            print(f"  Mask {i}: {mask_data['area']} pixels ({area_pct:.1f}%)")

        # Visualize all masks with colors
        visualization = sam.show_anns(masks)
        if visualization is not None:
            vis_img = Image.fromarray(visualization)
            vis_img.save(output_dir / "option2_sam_visualization.png")
            print(f"\n✓ Saved visualization with all {len(masks)} masks")

        del sam
        print("\n✓ SAM processing complete")

    except Exception as e:
        print(f"✗ SAM processing failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("Testing Alternative Segmentation Approaches")
    print("=" * 70)
    print()

    # Test Option 1
    test_option1_diff_to_rembg()

    # Test Option 2
    test_option2_sam_automatic()

    print()
    print("=" * 70)
    print("All tests complete! Check debug_diff_test/ for results:")
    print("  open debug_diff_test/")


if __name__ == "__main__":
    main()
