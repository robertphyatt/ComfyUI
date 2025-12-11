#!/usr/bin/env python3
"""Generate initial training masks using Claude's vision capabilities."""

import base64
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import anthropic


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def generate_mask_with_claude(base_frame_path: Path, clothed_frame_path: Path, client) -> np.ndarray:
    """Use Claude vision to generate a segmentation mask."""

    # Encode both images
    base_b64 = encode_image(base_frame_path)
    clothed_b64 = encode_image(clothed_frame_path)

    prompt = """You are analyzing pixel art sprite frames to identify clothing pixels.

I'm showing you TWO images:
1. BASE frame: Character without clothing (gray head, transparent body)
2. CLOTHED frame: Same character WITH clothing (brown armor added)

Your task: Identify which pixels are CLOTHING (the brown armor) vs BASE CHARACTER (gray head).

Return a JSON object with run-length encoding of the mask:
- 512x512 pixels total (262,144 pixels)
- Scan left-to-right, top-to-bottom
- 1 = clothing pixel (brown armor - KEEP THIS)
- 0 = non-clothing pixel (gray head, background - REMOVE THIS)

Format:
{
  "runs": [[value, count], [value, count], ...]
}

Example: [[0, 1000], [1, 50], [0, 500]] means:
- 1000 zeros (background/base)
- 50 ones (clothing)
- 500 zeros (more background)

Be precise: Only mark the brown armor pixels as 1. The gray head should be 0."""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base_b64
                    }
                },
                {
                    "type": "text",
                    "text": "BASE FRAME (character without clothing):"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": clothed_b64
                    }
                },
                {
                    "type": "text",
                    "text": "CLOTHED FRAME (character with clothing):"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )

    # Parse response
    response_text = message.content[0].text

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start:json_end]
        result = json.loads(json_str)

        # Decode run-length encoding
        mask = np.zeros(512 * 512, dtype=np.uint8)
        pos = 0
        for value, count in result["runs"]:
            mask[pos:pos+count] = value
            pos += count

        # Reshape to 512x512
        mask = mask.reshape(512, 512)
        return mask

    except Exception as e:
        print(f"  ✗ Failed to parse Claude response: {e}")
        print(f"  Response: {response_text[:200]}...")
        # Return empty mask on failure
        return np.zeros((512, 512), dtype=np.uint8)


def main():
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY environment variable not set")
        print("  Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    client = anthropic.Anthropic(api_key=api_key)

    frames_dir = Path("training_data/frames")
    masks_dir = Path("training_data/masks_initial")
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("Generating initial masks using Claude vision...")
    print("=" * 70)

    for frame_num in range(25):
        base_path = frames_dir / f"base_frame_{frame_num:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_num:02d}.png"
        mask_path = masks_dir / f"mask_{frame_num:02d}.png"

        if mask_path.exists():
            print(f"Frame {frame_num:02d}: Skipping (mask already exists)")
            continue

        print(f"Frame {frame_num:02d}: Analyzing with Claude vision...")

        try:
            mask = generate_mask_with_claude(base_path, clothed_path, client)

            # Save mask
            mask_img = Image.fromarray(mask * 255, 'L')
            mask_img.save(mask_path)

            # Statistics
            clothing_pixels = np.sum(mask == 1)
            percent = 100 * clothing_pixels / (512 * 512)
            print(f"  ✓ Generated mask: {clothing_pixels} clothing pixels ({percent:.1f}%)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print(f"✓ Initial masks saved to {masks_dir}/")
    print()
    print("Next step: Review and correct masks using the validation tool")


if __name__ == "__main__":
    main()
