#!/usr/bin/env python3
"""Test the hybrid AI approach on frame 12."""

from pathlib import Path
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_clothing_ai import extract_clothing_with_ai

def main():
    """Test on frame 12."""
    # Load frame 12 debug files
    base_frame = Image.open("debug_frames/frame_12_base.png")
    aligned_clothed = Image.open("debug_frames/frame_12_aligned.png")

    print("=" * 80)
    print("TESTING HYBRID APPROACH ON FRAME 12")
    print("=" * 80)
    print()

    # User guidance
    user_guidance = "Only the gray head is visible and should be removed. The body, arms, and legs are completely covered by brown leather armor."

    # Extract clothing using hybrid approach
    clothing_frame = extract_clothing_with_ai(base_frame, aligned_clothed, user_guidance, frame_num=12)

    # Save result
    output_path = Path("debug_frames_ai/frame_12_clothing_hybrid.png")
    output_path.parent.mkdir(exist_ok=True)
    clothing_frame.save(output_path)

    print()
    print(f"✓ Saved result to {output_path}")
    print(f"✓ Saved mask visualization to debug_frames_ai/mask_visualization_12.png")

    return 0

if __name__ == '__main__':
    sys.exit(main())
