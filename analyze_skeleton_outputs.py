#!/usr/bin/env python3
"""Analyze OpenPose skeleton outputs to check detection quality."""

from pathlib import Path
from PIL import Image
import numpy as np
import json

def analyze_skeleton(skeleton_path: Path) -> dict:
    """Analyze skeleton detection quality."""
    img = Image.open(skeleton_path)

    # Convert to array
    arr = np.array(img)

    # Count non-black pixels (skeleton lines are white/colored)
    if len(arr.shape) == 3:
        # RGB image - check if any channel is non-zero
        non_black = np.any(arr > 10, axis=2)
    else:
        # Grayscale
        non_black = arr > 10

    non_black_count = np.sum(non_black)
    total_pixels = arr.shape[0] * arr.shape[1]
    coverage_percent = (non_black_count / total_pixels) * 100

    return {
        "path": str(skeleton_path),
        "size": img.size,
        "non_black_pixels": int(non_black_count),
        "total_pixels": total_pixels,
        "coverage_percent": coverage_percent,
        "file_size_bytes": skeleton_path.stat().st_size
    }

if __name__ == "__main__":
    debug_dir = Path("output/debug")

    # Map frame numbers: frame_00 = openpose_skeleton_00001
    frame_mapping = {
        0: "openpose_skeleton_00001_.png",   # base_frame_00
        4: "openpose_skeleton_00005_.png",   # base_frame_04
        12: "openpose_skeleton_00013_.png"   # base_frame_12
    }

    results = {}
    for frame_num, skeleton_file in frame_mapping.items():
        skeleton_path = debug_dir / skeleton_file
        if skeleton_path.exists():
            results[frame_num] = analyze_skeleton(skeleton_path)
            print(f"=== FRAME {frame_num:02d} SKELETON ({skeleton_file}) ===")
            print(json.dumps(results[frame_num], indent=2))
            print()
        else:
            print(f"WARNING: {skeleton_path} not found")

    # Compare detection quality
    print("=== DETECTION QUALITY COMPARISON ===")
    for frame_num in sorted(results.keys()):
        r = results[frame_num]
        print(f"Frame {frame_num:02d}: {r['non_black_pixels']:6d} pixels ({r['coverage_percent']:.2f}% coverage) - {r['file_size_bytes']:6d} bytes")
