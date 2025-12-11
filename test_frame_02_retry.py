#!/usr/bin/env python3
"""Test frame 2 with retry logic (previously only 503 pixels)."""

from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from extract_clothing_ai import extract_clothing_with_ai

def main():
    """Test frame 2 with retry logic."""
    # Load frame 01 (which is "Frame 2/25" in output)
    base_frame = Image.open("debug_frames_ai/frame_01_base.png")
    aligned_clothed = Image.open("debug_frames_ai/frame_01_aligned.png")

    print("=" * 80)
    print("TESTING FRAME 2 WITH RETRY LOGIC")
    print("Previous result: 503 pixels removed (head still visible)")
    print("Expected: Retry until >= 1,500 pixels removed")
    print("=" * 80)
    print()

    user_guidance = "Only the gray head is visible and should be removed. The body, arms, and legs are completely covered by brown leather armor."

    clothing_frame = extract_clothing_with_ai(base_frame, aligned_clothed, user_guidance, frame_num=2, max_retries=3)

    output_path = Path("debug_frames_ai/frame_02_test_retry.png")
    clothing_frame.save(output_path)

    print()
    print(f"✓ Saved result to {output_path}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
