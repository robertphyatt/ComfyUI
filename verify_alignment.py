#!/usr/bin/env python3
"""Create visual verification images to validate alignment quality."""

import numpy as np
from PIL import Image
from pathlib import Path


def create_side_by_side_comparison(frame_idx: int, output_dir: Path):
    """Create side-by-side comparison: base | aligned_clothed | overlay."""
    frames_dir = Path("training_data/frames")
    aligned_dir = Path("training_data/frames_aligned")

    # Load frames
    base = Image.open(frames_dir / f"base_frame_{frame_idx:02d}.png").convert('RGBA')
    aligned = Image.open(aligned_dir / f"clothed_frame_{frame_idx:02d}.png").convert('RGBA')

    # Create overlay (base + aligned clothed)
    overlay = Image.alpha_composite(base, aligned)

    # Create side-by-side
    width, height = base.size
    comparison = Image.new('RGBA', (width * 3, height), (255, 255, 255, 255))

    comparison.paste(base, (0, 0))
    comparison.paste(aligned, (width, 0))
    comparison.paste(overlay, (width * 2, 0))

    # Save
    output_path = output_dir / f"verify_frame_{frame_idx:02d}.png"
    comparison.save(output_path)
    return output_path


def create_difference_image(frame_idx: int, output_dir: Path):
    """Create difference image highlighting misaligned pixels."""
    frames_dir = Path("training_data/frames")
    aligned_dir = Path("training_data/frames_aligned")

    # Load frames
    base = np.array(Image.open(frames_dir / f"base_frame_{frame_idx:02d}.png").convert('RGBA'))
    aligned = np.array(Image.open(aligned_dir / f"clothed_frame_{frame_idx:02d}.png").convert('RGBA'))

    # Calculate difference
    diff = np.abs(base.astype(int) - aligned.astype(int))

    # Highlight differences in red
    highlight = np.zeros_like(base)
    mask = np.any(diff > 10, axis=2)  # Any channel differs by >10
    highlight[mask] = [255, 0, 0, 255]  # Red

    # Create visualization
    result = Image.fromarray(highlight)
    output_path = output_dir / f"diff_frame_{frame_idx:02d}.png"
    result.save(output_path)
    return output_path


def main():
    """Generate verification images for all frames."""
    output_dir = Path("training_data/alignment_verification")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("GENERATING ALIGNMENT VERIFICATION IMAGES")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        # Side-by-side comparison
        comparison_path = create_side_by_side_comparison(frame_idx, output_dir)
        print(f"  ✓ Comparison: {comparison_path}")

        # Difference highlighting
        diff_path = create_difference_image(frame_idx, output_dir)
        print(f"  ✓ Difference: {diff_path}")

    print()
    print("=" * 70)
    print(f"✓ Verification images saved to {output_dir}/")
    print("=" * 70)
    print()
    print("Verification files:")
    print(f"  - verify_frame_XX.png: Base | Aligned | Overlay")
    print(f"  - diff_frame_XX.png: Red highlights show differences")


if __name__ == "__main__":
    main()
