#!/usr/bin/env python3
"""Extract clothing layers using AI vision segmentation.

Uses Ollama with ministral-3 to semantically segment base character from clothing.
"""

import argparse
import sys
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import requests
from extract_clothing_segmentation import extract_clothing_semantic


def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def upscale_for_analysis(image: Image.Image, scale: int = 8) -> Image.Image:
    """Upscale image for better AI vision analysis.

    Uses nearest neighbor to preserve pixel art aesthetic.

    Args:
        image: Original image
        scale: Upscale factor

    Returns:
        Upscaled image
    """
    width, height = image.size
    return image.resize((width * scale, height * scale), Image.NEAREST)


def call_ollama_bounding_box(base_frame: Image.Image, clothed_frame: Image.Image,
                             user_guidance: str = None,
                             ai_bounding_guidance: str = None,
                             max_retries: int = 100) -> Dict[str, Any]:
    """PHASE 1: Call Ollama vision API to get bounding boxes for BASE regions.

    This is a simple task for the AI: just identify rectangular regions.
    We'll use algorithmic flood-fill within these regions in Phase 2.

    Args:
        base_frame: Base character frame
        clothed_frame: Clothed character frame
        user_guidance: User-provided guidance on what base parts are visible
        ai_bounding_guidance: Additional guidance for AI on bounding box constraints
        max_retries: Maximum retry attempts on timeout

    Returns:
        Dict with bounding boxes for BASE regions
    """
    print(f"   [PHASE 1] Getting bounding boxes from AI...")
    print(f"   Frame size: {base_frame.width}x{base_frame.height}")

    # Encode images
    base_b64 = encode_image_base64(base_frame)
    clothed_b64 = encode_image_base64(clothed_frame)

    # Build user guidance section
    guidance_section = ""
    if user_guidance:
        guidance_section = f"""

USER GUIDANCE (CRITICAL - FOLLOW THIS EXACTLY):
{user_guidance}
"""

    # Build AI bounding box guidance section
    ai_guidance_section = ""
    if ai_bounding_guidance:
        ai_guidance_section = f"""

AI BOUNDING BOX CONSTRAINTS (CRITICAL - FOLLOW THESE CONSTRAINTS):
{ai_bounding_guidance}
"""

    # Prepare prompt for bounding box detection
    prompt = f"""You are analyzing two pixel art sprite frames to identify BASE character regions.

IMAGE 1 (base): A gray colored character - the naked base character
IMAGE 2 (clothed): The same gray character now wearing NEW CLOTHING/ARMOR added by Ludo.ai

YOUR TASK: Identify rectangular bounding boxes around visible BASE character parts in IMAGE 2.

BASE parts are regions that exist in IMAGE 1 (gray character body) and are still visible in IMAGE 2.
CLOTHING parts are new regions that only exist in IMAGE 2 (new armor/clothing).

For each visible BASE region in IMAGE 2, provide a simple RECTANGULAR BOUNDING BOX.

The image is {base_frame.width}x{base_frame.height} pixels.
Coordinates: X from 0 to {base_frame.width - 1}, Y from 0 to {base_frame.height - 1}

Output as JSON:
{{
  "image1_description": "Brief description of base character",
  "image2_description": "Brief description of clothed character",
  "base_regions": [
    {{
      "name": "e.g., 'gray head'",
      "description": "What this BASE region is",
      "bounding_box": {{
        "x_min": 200,
        "y_min": 50,
        "x_max": 350,
        "y_max": 180
      }},
      "reasoning": "Why this is a BASE region that needs removal"
    }}
  ]
}}

CRITICAL: Provide SIMPLE RECTANGULAR BOXES, not precise boundaries.
- x_min/y_min = top-left corner of the rectangle
- x_max/y_max = bottom-right corner of the rectangle
- Draw these boxes AROUND visible BASE parts (like the gray head)
- Make the boxes SLIGHTLY LARGER than the region to ensure full coverage
- Add ~20-30 pixels of padding on each side to avoid missing edges
{ai_guidance_section}{guidance_section}
Output ONLY valid JSON, no other text."""

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "ministral-3:8b",
        "prompt": prompt,
        "images": [base_b64, clothed_b64],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0  # Deterministic responses for consistency
        }
    }

    for attempt in range(max_retries):
        try:
            print(f"   Calling Ollama vision API (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(url, json=payload, timeout=600)

            print(f"   Response status: {response.status_code}")
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "")

            try:
                bounding_data = json.loads(response_text)
                num_regions = len(bounding_data.get('base_regions', []))
                print(f"   ✓ Got {num_regions} bounding box(es)")
                return bounding_data
            except json.JSONDecodeError as e:
                print(f"   ✗ Failed to parse JSON response: {e}")
                if attempt < max_retries - 1:
                    print("   Retrying...")
                    time.sleep(2)
                    continue
                raise

        except requests.exceptions.Timeout:
            print(f"   ✗ Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                print("   Retrying...")
                time.sleep(2)
                continue
            raise

        except requests.exceptions.RequestException as e:
            print(f"   ✗ Request failed: {e}")
            if attempt < max_retries - 1:
                print("   Retrying...")
                time.sleep(2)
                continue
            raise

    raise RuntimeError(f"Failed after {max_retries} attempts")


def create_mask_from_hybrid(base_frame: Image.Image, clothed_frame: Image.Image,
                            bounding_data: Dict[str, Any],
                            frame_num: int = 0,
                            tolerance: int = 10) -> np.ndarray:
    """PHASE 2 & 3: Create body mask using hybrid bounding box + direct color sampling.

    Args:
        base_frame: Base character frame
        clothed_frame: Clothed character frame
        bounding_data: Bounding box data from Phase 1
        frame_num: Frame number for debug output
        tolerance: Color matching tolerance (0-255, default: 10)

    Returns:
        Boolean mask where True = base character (remove), False = clothing (keep)
    """
    print(f"   [PHASE 3] Creating mask using hybrid approach...")

    clothed_arr = np.array(clothed_frame)
    height, width = clothed_arr.shape[:2]
    body_mask = np.zeros((height, width), dtype=bool)

    # Show AI's analysis from Phase 1
    img1_desc = bounding_data.get("image1_description", "")
    img2_desc = bounding_data.get("image2_description", "")
    if img1_desc:
        print(f"   IMAGE 1: {img1_desc}")
    if img2_desc:
        print(f"   IMAGE 2: {img2_desc}")

    base_regions = bounding_data.get("base_regions", [])
    print(f"   Processing {len(base_regions)} BASE region(s)...")

    for region in base_regions:
        name = region.get("name", "unknown")
        description = region.get("description", "")
        reasoning = region.get("reasoning", "")
        bbox = region.get("bounding_box", {})

        print(f"   ❌ BASE region: {name}")
        print(f"      Description: {description}")
        print(f"      Reasoning: {reasoning}")

        # PHASE 2: Sample actual colors from base frame instead of asking AI
        # Validate and clamp bounding box
        x_min = max(0, min(int(bbox.get('x_min', 0)), width - 1))
        y_min = max(0, min(int(bbox.get('y_min', 0)), height - 1))
        x_max = max(0, min(int(bbox.get('x_max', width-1)), width - 1))
        y_max = max(0, min(int(bbox.get('y_max', height-1)), height - 1))

        print(f"      Bounding box (raw): ({bbox.get('x_min')},{bbox.get('y_min')}) to ({bbox.get('x_max')},{bbox.get('y_max')})")

        # Validate box is well-formed
        if x_min >= x_max or y_min >= y_max:
            print(f"      ⚠️  WARNING: Invalid bounding box - skipping region")
            continue

        box_width = x_max - x_min
        box_height = y_max - y_min

        if box_width < 5 or box_height < 5:
            print(f"      ⚠️  WARNING: Bounding box too small ({box_width}x{box_height}) - skipping region")
            continue

        # CRITICAL: Reject boxes that are too large (likely including body, not just head)
        # Head should be roughly 150x150 pixels max for 512x512 frame
        MAX_HEAD_SIZE = 200
        if box_width > MAX_HEAD_SIZE or box_height > MAX_HEAD_SIZE:
            print(f"      ⚠️  WARNING: Bounding box too large ({box_width}x{box_height}) - likely includes body!")
            print(f"      Clamping to maximum {MAX_HEAD_SIZE}x{MAX_HEAD_SIZE}")
            # Clamp from center
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            half_size = MAX_HEAD_SIZE // 2
            x_min = max(0, center_x - half_size)
            x_max = min(width - 1, center_x + half_size)
            y_min = max(0, center_y - half_size)
            y_max = min(height - 1, center_y + half_size)
            # Recompute after clamping
            box_width = x_max - x_min
            box_height = y_max - y_min

        print(f"      Bounding box (validated): ({x_min},{y_min}) to ({x_max},{y_max}) [{box_width}x{box_height}]")
        print(f"      [PHASE 2] Sampling colors from base frame in this region...")

        # Sample all unique colors from base frame in this bounding box
        base_arr = np.array(base_frame)
        base_colors_set = set()

        for y in range(max(0, y_min), min(height, y_max + 1)):
            for x in range(max(0, x_min), min(width, x_max + 1)):
                pixel = base_arr[y, x]
                # Skip transparent
                if pixel[3] == 0:
                    continue
                base_colors_set.add((int(pixel[0]), int(pixel[1]), int(pixel[2])))

        num_colors = len(base_colors_set)
        print(f"      Found {num_colors} unique colors in base frame")

        # Now remove any pixels in clothed frame that match these base colors
        # with tolerance for compression artifacts and lighting variations
        print(f"      Using tolerance: {tolerance}")

        pixels_removed = 0
        for y in range(max(0, y_min), min(height, y_max + 1)):
            for x in range(max(0, x_min), min(width, x_max + 1)):
                clothed_pixel = clothed_arr[y, x]

                # Skip if transparent
                if clothed_pixel[3] == 0:
                    continue

                c_r, c_g, c_b = int(clothed_pixel[0]), int(clothed_pixel[1]), int(clothed_pixel[2])

                # Check if this pixel matches any base color (with tolerance)
                for b_r, b_g, b_b in base_colors_set:
                    if (abs(c_r - b_r) <= tolerance and
                        abs(c_g - b_g) <= tolerance and
                        abs(c_b - b_b) <= tolerance):
                        body_mask[y, x] = True
                        pixels_removed += 1
                        break  # Found a match, move to next pixel

        print(f"      → Removed {pixels_removed} pixels in this region")

    # Save visualization
    mask_vis = np.zeros((height, width, 3), dtype=np.uint8)
    mask_vis[body_mask] = [255, 0, 0]  # Red for removed pixels

    # Draw bounding boxes in green
    for region in base_regions:
        bbox = region.get("bounding_box", {})
        x_min = bbox.get('x_min', 0)
        y_min = bbox.get('y_min', 0)
        x_max = bbox.get('x_max', width-1)
        y_max = bbox.get('y_max', height-1)

        # Draw rectangle
        if y_min < height and x_min < width:
            mask_vis[y_min, x_min:min(x_max+1, width)] = [0, 255, 0]  # Top
        if y_max < height and x_min < width:
            mask_vis[y_max, x_min:min(x_max+1, width)] = [0, 255, 0]  # Bottom
        if x_min < width and y_min < height:
            mask_vis[y_min:min(y_max+1, height), x_min] = [0, 255, 0]  # Left
        if x_max < width and y_min < height:
            mask_vis[y_min:min(y_max+1, height), x_max] = [0, 255, 0]  # Right

    from PIL import Image as PILImage
    mask_path = f"debug_frames_ai/mask_visualization_{frame_num:02d}.png"
    PILImage.fromarray(mask_vis).save(mask_path)
    print(f"      Saved mask visualization to {mask_path}")

    total_removed = np.sum(body_mask)
    print(f"   Total pixels removed: {total_removed}")

    return body_mask


def extract_clothing_with_ai(base_frame: Image.Image, clothed_frame: Image.Image,
                            user_guidance: str = None, ai_bounding_guidance: str = None,
                            frame_num: int = 0,
                            tolerance: int = 10, max_retries: int = 3,
                            return_pixel_count: bool = False):
    """Extract clothing using hybrid AI + algorithmic approach.

    Three-phase approach:
    1. AI identifies bounding boxes around BASE regions (semantic understanding)
    2. Direct color sampling from base frame within bounding boxes (algorithmic precision)
    3. Tolerance-based color matching for pixel removal (handles compression artifacts)

    Args:
        base_frame: Base character frame
        clothed_frame: Clothed character frame
        user_guidance: User-provided guidance on what base parts to remove
        ai_bounding_guidance: Additional guidance for AI on bounding box constraints
        frame_num: Frame number for debug output
        tolerance: Color matching tolerance (0-255, default: 10)
        max_retries: Maximum retry attempts if result is suspicious (default: 3)

    Returns:
        Clothing-only frame with transparent background
    """
    # Ensure RGBA
    if base_frame.mode != 'RGBA':
        base_frame = base_frame.convert('RGBA')
    if clothed_frame.mode != 'RGBA':
        clothed_frame = clothed_frame.convert('RGBA')

    # Retry loop for inconsistent AI bounding boxes
    MIN_PIXELS_REMOVED = 1500  # Head should be at least 1500 pixels

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"\n   🔄 Retry attempt {attempt + 1}/{max_retries} (previous result was suspicious)")

        # PHASE 1: Get bounding boxes from AI
        bounding_data = call_ollama_bounding_box(base_frame, clothed_frame, user_guidance, ai_bounding_guidance)

        # PHASE 2 & 3: Sample colors and create mask (combined in create_mask_from_hybrid)
        body_mask = create_mask_from_hybrid(base_frame, clothed_frame, bounding_data, frame_num, tolerance)

        # Count results
        clothed_arr = np.array(clothed_frame)
        body_pixels = np.sum(body_mask)
        total_clothed = np.sum(clothed_arr[:, :, 3] > 0)
        clothing_pixels = total_clothed - body_pixels

        print(f"   ✓ Attempt {attempt + 1}: kept {clothing_pixels} clothing pixels, removed {body_pixels} body pixels")

        # Check if result is reasonable
        if body_pixels >= MIN_PIXELS_REMOVED:
            print(f"   ✅ Result looks good (>= {MIN_PIXELS_REMOVED} pixels removed)")
            break
        else:
            print(f"   ⚠️  Suspicious result: only {body_pixels} pixels removed (expected >= {MIN_PIXELS_REMOVED})")
            if attempt < max_retries - 1:
                print(f"   🔄 Retrying with fresh AI call...")
                continue
            else:
                print(f"   ⚠️  Max retries reached, proceeding with best attempt")

    # Apply mask to extract clothing
    clothing_arr = clothed_arr.copy()
    clothing_arr[body_mask] = [0, 0, 0, 0]

    result = Image.fromarray(clothing_arr, 'RGBA')

    if return_pixel_count:
        return result, body_pixels
    else:
        return result


def split_spritesheet(spritesheet_path: Path, grid_size: Tuple[int, int] = (5, 5)) -> List[Image.Image]:
    """Split a spritesheet into individual frames."""
    img = Image.open(spritesheet_path)
    width, height = img.size
    cols, rows = grid_size

    frame_width = width // cols
    frame_height = height // rows

    frames = []
    for row in range(rows):
        for col in range(cols):
            left = col * frame_width
            top = row * frame_height
            right = left + frame_width
            bottom = top + frame_height

            frame = img.crop((left, top, right, bottom))
            frames.append(frame)

    return frames


def find_character_bounds(frame: Image.Image) -> Tuple[int, int, int, int]:
    """Find bounding box of non-transparent pixels in frame."""
    if frame.mode != 'RGBA':
        frame = frame.convert('RGBA')

    alpha = np.array(frame)[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    return (left, top, right, bottom)


def align_frames(base_frame: Image.Image, clothed_frame: Image.Image) -> Image.Image:
    """Align clothed frame to match base frame position."""
    base_bounds = find_character_bounds(base_frame)
    clothed_bounds = find_character_bounds(clothed_frame)

    if base_bounds is None or clothed_bounds is None:
        return clothed_frame

    base_left, base_top, base_right, base_bottom = base_bounds
    clothed_left, clothed_top, clothed_right, clothed_bottom = clothed_bounds

    # Calculate offset to align centers
    base_center_x = (base_left + base_right) // 2
    base_center_y = (base_top + base_bottom) // 2
    clothed_center_x = (clothed_left + clothed_right) // 2
    clothed_center_y = (clothed_top + clothed_bottom) // 2

    offset_x = base_center_x - clothed_center_x
    offset_y = base_center_y - clothed_center_y

    # Create new image with same size
    aligned = Image.new('RGBA', clothed_frame.size, (0, 0, 0, 0))
    aligned.paste(clothed_frame, (offset_x, offset_y), clothed_frame)

    return aligned


def reassemble_spritesheet(frames: List[Image.Image], output_path: Path,
                          grid_size: Tuple[int, int] = (5, 5)) -> Path:
    """Reassemble frames into a spritesheet."""
    cols, rows = grid_size
    frame_width = frames[0].width
    frame_height = frames[0].height

    # Create output image
    output = Image.new('RGBA', (frame_width * cols, frame_height * rows), (0, 0, 0, 0))

    # Paste each frame
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * frame_width
        y = row * frame_height
        output.paste(frame, (x, y), frame)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract clothing layers using AI vision segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python extract_clothing_ai.py \\
        --base examples/input/base.png \\
        --clothed examples/input/reference.png \\
        --output examples/output/clothing_ai.png
        """
    )

    parser.add_argument(
        '--base',
        type=Path,
        required=True,
        help='Base spritesheet (naked character)'
    )

    parser.add_argument(
        '--clothed',
        type=Path,
        required=True,
        help='Clothed spritesheet (character wearing clothing)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output/clothing_ai.png'),
        help='Output path for clothing-only spritesheet (default: output/clothing_ai.png)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save intermediate frames for debugging'
    )

    parser.add_argument(
        '--guidance',
        type=str,
        default=None,
        help='User guidance on which base parts to remove (e.g., "Only the gray head is visible and should be removed")'
    )

    parser.add_argument(
        '--ai-guidance',
        type=str,
        default=None,
        help='AI-specific guidance for bounding box constraints (e.g., "Only draw boxes around the HEAD. Exclude shoulders, neck, and torso.")'
    )

    parser.add_argument(
        '--tolerance',
        type=int,
        default=10,
        help='Color matching tolerance for compression artifacts (default: 10)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['bbox', 'semantic'],
        default='bbox',
        help='Extraction method: bbox (bounding box with guidance) or semantic (pixel-level segmentation with edge refinement). Default: bbox'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.base.exists():
        print(f"Error: Base spritesheet not found: {args.base}", file=sys.stderr)
        return 1

    if not args.clothed.exists():
        print(f"Error: Clothed spritesheet not found: {args.clothed}", file=sys.stderr)
        return 1

    print("🤖 Extracting clothing layers using AI vision...")
    print(f"   Base: {args.base}")
    print(f"   Clothed: {args.clothed}")
    print(f"   Method: {args.method}")
    if args.guidance:
        print(f"   Guidance: {args.guidance}")
    if args.ai_guidance:
        print(f"   AI Guidance: {args.ai_guidance}")

    # Check Ollama availability
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("   ✓ Ollama API is available")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Ollama API not available: {e}", file=sys.stderr)
        print("   Make sure Ollama is running with: ollama serve", file=sys.stderr)
        return 1

    # Step 1: Split spritesheets
    print("\n📦 Step 1: Splitting spritesheets...")
    base_frames = split_spritesheet(args.base)
    clothed_frames = split_spritesheet(args.clothed)
    print(f"   Split into {len(base_frames)} frames each")

    # Step 2: Process each frame with AI (two-pass adaptive retry)
    print("\n🔧 Step 2: Processing frames with AI vision...")
    clothing_frames = [None] * len(base_frames)
    pixel_counts = [0] * len(base_frames)

    if args.debug:
        debug_dir = Path("debug_frames_ai")
        debug_dir.mkdir(exist_ok=True)

    if args.method == 'bbox':
        # EXISTING BBOX LOGIC (two-pass adaptive retry)
        # PASS 1: Initial processing
        print("\n   === PASS 1: Initial processing ===")
        for i, (base_frame, clothed_frame) in enumerate(zip(base_frames, clothed_frames)):
            print(f"\n   Frame {i+1}/{len(base_frames)}:")

            # Align clothed frame to base
            aligned_clothed = align_frames(base_frame, clothed_frame)

            # Extract clothing using AI (no retry on first pass)
            clothing_frame, pixels_removed = extract_clothing_with_ai(
                base_frame, aligned_clothed, args.guidance, args.ai_guidance, i,
                tolerance=args.tolerance, max_retries=1, return_pixel_count=True
            )
            clothing_frames[i] = clothing_frame
            pixel_counts[i] = pixels_removed

            # Save debug frames if requested
            if args.debug:
                base_frame.save(debug_dir / f"frame_{i:02d}_base.png")
                clothed_frame.save(debug_dir / f"frame_{i:02d}_clothed.png")
                aligned_clothed.save(debug_dir / f"frame_{i:02d}_aligned.png")
                clothing_frame.save(debug_dir / f"frame_{i:02d}_clothing.png")

        # PASS 2: Adaptive retry until all frames within 10% of max
        print(f"\n   === PASS 2: Adaptive retry ===")
        max_retries_per_frame = 5
        retry_round = 1

        while True:
            max_pixels = max(pixel_counts)
            min_threshold = max_pixels * 0.10  # 10% of max

            retry_indices = [i for i, count in enumerate(pixel_counts) if count < min_threshold]

            if not retry_indices:
                print(f"   ✅ All frames within 10% of max ({max_pixels} pixels)")
                break

            print(f"\n   Round {retry_round}: {len(retry_indices)} frames below threshold ({min_threshold:.0f} pixels)")
            print(f"   Frames to retry: {[i+1 for i in retry_indices]}")

            improved = False
            for i in retry_indices:
                print(f"\n   Retrying frame {i+1} (currently {pixel_counts[i]} pixels):")

                base_frame = base_frames[i]
                clothed_frame = clothed_frames[i]
                aligned_clothed = align_frames(base_frame, clothed_frame)

                # Retry with fresh AI call
                clothing_frame, pixels_removed = extract_clothing_with_ai(
                    base_frame, aligned_clothed, args.guidance, args.ai_guidance, i,
                    tolerance=args.tolerance, max_retries=1, return_pixel_count=True
                )

                if pixels_removed > pixel_counts[i]:
                    print(f"   ✅ Improved: {pixel_counts[i]} → {pixels_removed} pixels")
                    clothing_frames[i] = clothing_frame
                    pixel_counts[i] = pixels_removed
                    improved = True

                    # Update debug frame
                    if args.debug:
                        clothing_frame.save(debug_dir / f"frame_{i:02d}_clothing.png")
                else:
                    print(f"   ℹ️  No improvement ({pixels_removed} pixels)")

            if not improved:
                print(f"\n   ⚠️  No improvements in round {retry_round}, stopping retries")
                print(f"   Final pixel counts range: {min(pixel_counts)} - {max(pixel_counts)}")
                break

            retry_round += 1
            if retry_round > max_retries_per_frame:
                print(f"\n   ⚠️  Max retry rounds reached ({max_retries_per_frame}), stopping")
                break

    elif args.method == 'semantic':
        # NEW SEMANTIC SEGMENTATION APPROACH (no adaptive retry needed)
        print("\n   === Semantic Segmentation Processing ===")
        for i, (base_frame, clothed_frame) in enumerate(zip(base_frames, clothed_frames)):
            print(f"\n   Frame {i+1}/{len(base_frames)}:")

            # No alignment needed - semantic segmentation handles variations
            # Extract clothing using semantic segmentation
            start_time = time.time()
            clothing_frame = extract_clothing_semantic(clothed_frame, base_frame)
            elapsed = time.time() - start_time

            clothing_frames[i] = clothing_frame
            print(f"   ✓ Completed in {elapsed:.2f}s")

            # Save debug frames if requested
            if args.debug:
                debug_dir_semantic = Path("debug_frames_semantic")
                debug_dir_semantic.mkdir(exist_ok=True)
                base_frame.save(debug_dir_semantic / f"frame_{i:02d}_base.png")
                clothed_frame.save(debug_dir_semantic / f"frame_{i:02d}_clothed.png")
                clothing_frame.save(debug_dir_semantic / f"frame_{i:02d}_clothing.png")

    # Step 3: Reassemble
    print("\n🔧 Step 3: Reassembling spritesheet...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reassemble_spritesheet(clothing_frames, args.output)
    print(f"   Saved to {args.output}")

    if args.debug:
        print(f"\n🐛 Debug frames saved to {debug_dir}/")

    print("\n✅ AI extraction complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
