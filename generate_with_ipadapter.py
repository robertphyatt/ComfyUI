#!/usr/bin/env python3
"""Generate clothed frames using IPAdapter + ControlNet."""

import sys
from pathlib import Path
from PIL import Image
from sprite_clothing_gen.comfy_client import ComfyUIClient
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow


def generate_clothed_frame(client: ComfyUIClient, frame_idx: int,
                          frames_dir: Path, masks_dir: Path,
                          output_dir: Path) -> bool:
    """Generate one clothed frame using IPAdapter.

    Args:
        client: ComfyUI client
        frame_idx: Frame index (0-24)
        frames_dir: Directory with base frames
        masks_dir: Directory with inpainting masks
        output_dir: Directory for generated frames

    Returns:
        True if successful
    """
    base_name = f"base_frame_{frame_idx:02d}.png"
    mask_name = f"mask_{frame_idx:02d}.png"

    # Upload base image and mask
    base_path = frames_dir / base_name
    mask_path = masks_dir / mask_name

    if not base_path.exists():
        print(f"  ✗ Base frame not found: {base_path}")
        return False

    if not mask_path.exists():
        print(f"  ✗ Mask not found: {mask_path}")
        return False

    print(f"  Uploading base and mask...")
    client.upload_image(base_path)
    client.upload_image(mask_path)

    # Build workflow
    print(f"  Building workflow...")

    # Reference images (all 25 clothed frames)
    reference_names = [f"clothed_frame_{i:02d}.png" for i in range(25)]

    workflow = build_ipadapter_generation_workflow(
        base_image_name=base_name,
        mask_image_name=mask_name,
        reference_image_names=reference_names,
        prompt="Brown leather armor with shoulder pauldrons, chest plate, arm guards, leg armor, fantasy RPG character, pixel art style, detailed, high quality",
        negative_prompt="blurry, low quality, distorted, deformed, multiple heads, extra limbs, modern clothing",
        seed=42 + frame_idx,  # Different seed per frame
        steps=35,
        cfg=7.0,
        denoise=1.0
    )

    # Queue and wait
    print(f"  Generating with IPAdapter...")
    prompt_id = client.queue_prompt(workflow)

    try:
        history = client.wait_for_completion(prompt_id, timeout=180)
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        return False

    # Get output image
    outputs = history.get('outputs', {})
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            # Download generated image
            for img_info in node_output['images']:
                filename = img_info['filename']
                subfolder = img_info.get('subfolder', '')

                # Download the image
                output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
                client.download_image(filename, subfolder, output_dir=output_dir)

                # Rename to expected name if needed
                downloaded_path = output_dir / filename
                if downloaded_path != output_path:
                    downloaded_path.rename(output_path)

                print(f"  ✓ Saved to {output_path}")
                return True

    print(f"  ✗ No output image found")
    return False


def main():
    """Generate all 25 clothed frames using IPAdapter."""
    # Check ComfyUI is running
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        print("ERROR: ComfyUI server not running at http://127.0.0.1:8188")
        print("Start it with: cd /Users/roberthyatt/Code/ComfyUI && python main.py")
        return 1

    frames_dir = Path("training_data/frames")
    masks_dir = Path("training_data/masks_inpainting")
    output_dir = Path("training_data/frames_ipadapter_generated")
    output_dir.mkdir(exist_ok=True)

    # Upload all 25 reference frames first
    print("=" * 70)
    print("UPLOADING REFERENCE FRAMES")
    print("=" * 70)
    print()

    for i in range(25):
        ref_path = frames_dir / f"clothed_frame_{i:02d}.png"
        if ref_path.exists():
            client.upload_image(ref_path)
            print(f"Uploaded reference frame {i:02d}")

    print()
    print("=" * 70)
    print("GENERATING CLOTHED FRAMES WITH IPADAPTER + CONTROLNET")
    print("=" * 70)
    print()

    success_count = 0

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        if generate_clothed_frame(client, frame_idx, frames_dir, masks_dir, output_dir):
            success_count += 1

        print()

    print("=" * 70)
    print(f"✓ Generated {success_count}/25 frames")
    print("=" * 70)

    return 0 if success_count == 25 else 1


if __name__ == "__main__":
    sys.exit(main())
