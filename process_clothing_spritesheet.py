#!/usr/bin/env python3
"""Unified pipeline for clothing spritesheet generation.

Runs complete workflow:
1. Align clothed frames to base using OpenPose
2. Extend armor to cover feet
3. Open mask validation tool for user review
4. Extract final clothing spritesheet

Usage:
    python process_clothing_spritesheet.py
"""

import sys
from pathlib import Path
from align_with_openpose import main as align_main
from extend_armor_feet import main as extend_main
from validate_predicted_masks import main as validate_main
from extract_clothing_final import main as extract_main


def main():
    """Run complete clothing spritesheet generation pipeline."""
    print("=" * 70)
    print("CLOTHING SPRITESHEET GENERATION PIPELINE")
    print("=" * 70)
    print()
    print("This pipeline will:")
    print("  1. Align clothed frames to base using OpenPose skeleton matching")
    print("  2. Extend armor to cover feet")
    print("  3. Generate initial masks using trained U-Net model")
    print("  4. Open mask validation tool for manual review")
    print("  5. Extract final clothing spritesheet from validated masks")
    print()
    print("=" * 70)
    print()

    # Step 1: Align frames using OpenPose
    print("\n" + "=" * 70)
    print("STEP 1/5: Aligning frames with OpenPose")
    print("=" * 70 + "\n")

    result = align_main()
    if result != 0:
        print("ERROR: OpenPose alignment failed")
        return 1

    # Step 2: Extend armor to cover feet
    print("\n" + "=" * 70)
    print("STEP 2/5: Extending armor to cover feet")
    print("=" * 70 + "\n")

    result = extend_main()
    if result != 0:
        print("ERROR: Armor extension failed")
        return 1

    # Step 3: Generate initial masks with U-Net
    print("\n" + "=" * 70)
    print("STEP 3/5: Generating masks with trained U-Net model")
    print("=" * 70 + "\n")

    import subprocess
    result = subprocess.run([
        sys.executable,
        "predict_masks_with_model.py",
        "--frames-dir", "training_data/frames_complete_openpose",
        "--output-dir", "training_data_validation/masks_corrected"
    ])

    if result.returncode != 0:
        print("ERROR: Mask prediction failed")
        return 1

    # Step 4: Open mask validation tool
    print("\n" + "=" * 70)
    print("STEP 4/5: Opening mask validation tool")
    print("=" * 70 + "\n")
    print("Review and correct masks as needed...")
    print("Press Save to accept each mask and move to next frame")
    print()

    # Copy complete frames to validation directory
    import shutil
    val_frames = Path("training_data_validation/frames")
    val_frames.mkdir(parents=True, exist_ok=True)

    for i in range(25):
        src = Path(f"training_data/frames_complete_openpose/clothed_frame_{i:02d}.png")
        dst = val_frames / f"clothed_frame_{i:02d}.png"
        shutil.copy(src, dst)

    result = validate_main()
    if result != 0:
        print("ERROR: Mask validation failed or cancelled")
        return 1

    # Step 5: Extract final clothing spritesheet
    print("\n" + "=" * 70)
    print("STEP 5/5: Extracting final clothing spritesheet")
    print("=" * 70 + "\n")

    # Update extraction to use validated masks
    from extract_clothing_final import extract_with_validated_masks

    output_path = Path("training_data/clothing_spritesheet_final.png")
    extract_with_validated_masks(
        frames_dir=Path("training_data/frames_complete_openpose"),
        masks_dir=Path("training_data_validation/masks_corrected"),
        output_path=output_path
    )

    # Create final verification overlay
    from PIL import Image
    base = Image.open("examples/input/base.png").convert('RGBA')
    clothing = Image.open(output_path).convert('RGBA')
    overlay = Image.alpha_composite(base, clothing)
    overlay.save("training_data/final_verification.png")

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("Final deliverables:")
    print(f"  - Clothing spritesheet: {output_path}")
    print(f"  - Verification overlay: training_data/final_verification.png")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
