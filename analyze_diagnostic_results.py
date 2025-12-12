#!/usr/bin/env python3
"""Analyze the OpenPose diagnostic results."""

from pathlib import Path
from PIL import Image
import numpy as np
import json

def analyze_skeleton_detection(image_path: Path) -> dict:
    """Analyze if skeleton was detected."""
    img = Image.open(image_path)
    arr = np.array(img)

    # Count non-black pixels
    if len(arr.shape) == 3:
        non_black = np.any(arr > 10, axis=2)
    else:
        non_black = arr > 10

    non_black_count = int(np.sum(non_black))
    total_pixels = arr.shape[0] * arr.shape[1]

    return {
        "path": str(image_path.name),
        "size_bytes": image_path.stat().st_size,
        "dimensions": img.size,
        "non_black_pixels": non_black_count,
        "coverage_percent": (non_black_count / total_pixels) * 100,
        "detected": non_black_count > 100  # Threshold for "detected"
    }

if __name__ == "__main__":
    diagnostic_dir = Path("output/openpose_diagnostic")

    # Analyze all three test approaches for each frame
    variants = ["default", "body_only", "highres"]
    frames = [0, 4, 12]

    results = {}
    for frame in frames:
        results[frame] = {}
        print(f"=== FRAME {frame:02d} ===")
        for variant in variants:
            pattern = f"frame_{frame:02d}_openpose_{variant}_*.png"
            files = list(diagnostic_dir.glob(pattern))
            if files:
                result = analyze_skeleton_detection(files[0])
                results[frame][variant] = result
                detected_str = "✓ DETECTED" if result["detected"] else "✗ NO DETECTION"
                print(f"  {variant:12s}: {result['non_black_pixels']:6d} pixels ({result['coverage_percent']:.2f}%) - {detected_str}")
        print()

    # Summary
    print("=== SUMMARY ===")
    for frame in frames:
        detected_count = sum(1 for v in results[frame].values() if v["detected"])
        status = "WORKING" if detected_count > 0 else "FAILING"
        print(f"Frame {frame:02d}: {status} ({detected_count}/3 variants detected)")

    # Save results
    output_file = Path("diagnostic_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
