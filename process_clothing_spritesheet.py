#!/usr/bin/env python3
"""Unified pipeline for clothing spritesheet generation.

Runs complete workflow:
1. Start ComfyUI server
2. Generate inpainting masks from alpha channel
3. Generate clothed frames using IPAdapter + ControlNet
4. Extend armor to cover feet
5. Generate initial masks using trained U-Net model
6. Open mask validation tool for manual review
7. Extract final clothing spritesheet from validated masks
8. Stop ComfyUI server

Usage:
    python process_clothing_spritesheet.py           # Clean start (default)
    python process_clothing_spritesheet.py --resume  # Keep existing artifacts
"""

import sys
import subprocess
import argparse
import shutil
import time
import signal
import atexit
from pathlib import Path
from generate_with_ipadapter import main as ipadapter_main
from generate_inpainting_masks import main as mask_gen_main
from extend_armor_feet import main as extend_main
from validate_predicted_masks import main as validate_main
from extract_clothing_final import main as extract_main
from extract_training_frames import main as extract_frames_main

# Global variable to track ComfyUI process
_comfyui_process = None


def start_comfyui_server():
    """Start ComfyUI server in background."""
    global _comfyui_process

    print("Starting ComfyUI server...")
    print("(This may take 30-60 seconds)")
    print()

    # Start server in background, redirecting output to log
    log_file = open("comfyui_server.log", "w")
    _comfyui_process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent)
    )

    # Wait for server to be ready
    from sprite_clothing_gen.comfy_client import ComfyUIClient
    client = ComfyUIClient("http://127.0.0.1:8188")

    max_wait = 120  # 2 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if client.health_check():
            print("✓ ComfyUI server is ready")
            print()
            return True
        time.sleep(2)

    print("ERROR: ComfyUI server failed to start within 2 minutes")
    return False


def stop_comfyui_server():
    """Stop ComfyUI server."""
    global _comfyui_process

    if _comfyui_process is not None:
        print()
        print("Stopping ComfyUI server...")

        # Send SIGTERM to gracefully shut down
        _comfyui_process.terminate()

        try:
            # Wait up to 10 seconds for graceful shutdown
            _comfyui_process.wait(timeout=10)
            print("✓ ComfyUI server stopped")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            _comfyui_process.kill()
            _comfyui_process.wait()
            print("✓ ComfyUI server killed (forced)")

        _comfyui_process = None


# Register cleanup handler
atexit.register(stop_comfyui_server)


def clean_artifacts():
    """Remove all pipeline artifacts for a fresh start."""
    print("Cleaning previous artifacts...")

    artifacts = [
        Path("training_data/frames_ipadapter_generated"),
        Path("output/debug"),
    ]

    # Remove directories
    for path in artifacts:
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed {path}")

    # Remove ipadapter output files
    output_dir = Path("output")
    if output_dir.exists():
        for f in output_dir.glob("ipadapter_generated_*.png"):
            f.unlink()
            print(f"  Removed {f}")

    print("  ✓ Artifacts cleaned")
    print()


def main():
    """Run complete clothing spritesheet generation pipeline."""
    parser = argparse.ArgumentParser(description="Clothing spritesheet generation pipeline")
    parser.add_argument("--resume", action="store_true",
                        help="Keep existing artifacts instead of cleaning")
    args = parser.parse_args()

    if not args.resume:
        clean_artifacts()
    else:
        print("Resuming with existing artifacts...")
        print()

    return run_pipeline()


def run_pipeline():
    """Run complete clothing spritesheet generation pipeline."""
    print("=" * 70)
    print("CLOTHING SPRITESHEET GENERATION PIPELINE")
    print("=" * 70)
    print()
    print("This pipeline will:")
    print("  1. Extract frames from reference spritesheets")
    print("  2. Start ComfyUI server")
    print("  3. Generate inpainting masks")
    print("  4. Generate clothed frames using IPAdapter + ControlNet")
    print("  5. Extend armor to cover feet")
    print("  6. Generate initial masks using trained U-Net model")
    print("  7. Open mask validation tool for manual review")
    print("  8. Extract final clothing spritesheet from validated masks")
    print("  9. Stop ComfyUI server")
    print()
    print("=" * 70)
    print()

    # Step 1: Extract frames from spritesheets
    print("\n" + "=" * 70)
    print("STEP 1/8: Extracting frames from reference spritesheets")
    print("=" * 70 + "\n")

    result = extract_frames_main()
    if result != 0:
        print("ERROR: Frame extraction failed")
        return 1

    # Start ComfyUI server
    if not start_comfyui_server():
        print("ERROR: Failed to start ComfyUI server")
        return 1

    try:
        # Step 2: Generate inpainting masks
        print("\n" + "=" * 70)
        print("STEP 2/8: Generating inpainting masks")
        print("=" * 70 + "\n")

        result = mask_gen_main()
        if result != 0:
            print("ERROR: Mask generation failed")
            return 1

        # Step 3: Generate clothed frames with IPAdapter
        print("\n" + "=" * 70)
        print("STEP 3/8: Generating with IPAdapter + ControlNet")
        print("=" * 70 + "\n")

        result = ipadapter_main()
        if result != 0:
            print("ERROR: IPAdapter generation failed")
            return 1

        # Step 4: Extend armor to cover feet
        print("\n" + "=" * 70)
        print("STEP 4/8: Extending armor to cover feet")
        print("=" * 70 + "\n")

        result = extend_main()
        if result != 0:
            print("ERROR: Armor extension failed")
            return 1

        # Step 5: Generate initial masks with U-Net
        print("\n" + "=" * 70)
        print("STEP 5/8: Generating masks with trained U-Net model")
        print("=" * 70 + "\n")

        import subprocess
        result = subprocess.run([
            sys.executable,
            "predict_masks_with_model.py",
            "--frames-dir", "training_data/frames_complete_ipadapter",
            "--output-dir", "training_data_validation/masks_corrected"
        ])

        if result.returncode != 0:
            print("ERROR: Mask prediction failed")
            return 1

        # Step 6: Open mask validation tool
        print("\n" + "=" * 70)
        print("STEP 6/8: Opening mask validation tool")
        print("=" * 70 + "\n")
        print("Review and correct masks as needed...")
        print("Press Save to accept each mask and move to next frame")
        print()

        # Copy complete frames to validation directory
        import shutil
        val_frames = Path("training_data_validation/frames")
        val_frames.mkdir(parents=True, exist_ok=True)

        for i in range(25):
            src = Path(f"training_data/frames_complete_ipadapter/clothed_frame_{i:02d}.png")
            dst = val_frames / f"clothed_frame_{i:02d}.png"
            shutil.copy(src, dst)

        result = validate_main()
        if result != 0:
            print("ERROR: Mask validation failed or cancelled")
            return 1

        # Step 7: Extract final clothing spritesheet
        print("\n" + "=" * 70)
        print("STEP 7/8: Extracting final clothing spritesheet")
        print("=" * 70 + "\n")

        # Update extraction to use validated masks
        from extract_clothing_final import extract_with_validated_masks

        output_path = Path("training_data/clothing_spritesheet_ipadapter.png")
        extract_with_validated_masks(
            frames_dir=Path("training_data/frames_complete_ipadapter"),
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

    finally:
        # Step 8: Stop server
        print("\n" + "=" * 70)
        print("STEP 8/8: Stopping ComfyUI server")
        print("=" * 70 + "\n")
        stop_comfyui_server()


if __name__ == "__main__":
    sys.exit(main())
