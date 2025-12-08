#!/usr/bin/env python3
"""Test frame 1 which was problematic (removed 7,635 pixels)."""

from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from extract_clothing_ai import extract_clothing_with_ai

def main():
    """Test on frame 1."""
    # Load frame 0 (which is "Frame 1/25" in output - 1-indexed)
    base_frame = Image.open("debug_frames_ai/frame_00_base.png")
    aligned_clothed = Image.open("debug_frames_ai/frame_00_aligned.png")

    print("=" * 80)
    print("TESTING FRAME 1 (Previously removed 7,635 pixels - TOO MUCH)")
    print("=" * 80)
    print()

    user_guidance = "Only the gray head is visible and should be removed. The body, arms, and legs are completely covered by brown leather armor."

    clothing_frame = extract_clothing_with_ai(base_frame, aligned_clothed, user_guidance, frame_num=1)

    output_path = Path("debug_frames_ai/frame_01_test_validated.png")
    clothing_frame.save(output_path)

    print()
    print(f"✓ Saved result to {output_path}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
