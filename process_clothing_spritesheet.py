#!/usr/bin/env python3
"""DEPRECATED: This pipeline uses IPAdapter which cannot preserve exact textures.

Use the optical flow pipeline instead:

    python generate_sprite_clothing_optical.py \\
        --base input/mannequin_spritesheet.png \\
        --clothed input/clothed_spritesheet.png \\
        --output output/result_spritesheet.png

The optical flow approach:
  - Preserves EXACT armor textures (pixel-perfect)
  - Runs in <1s per frame on CPU (no GPU needed)
  - Deterministic output (same input = same output)
  - Skips warping when poses already match

See docs/findings/2025-12-13-depth-controlnet-orientation-fix.md for details
on why AI-based approaches (IPAdapter, ControlNet) fail to preserve textures.
"""

import sys


def main():
    print()
    print("=" * 70)
    print("DEPRECATED PIPELINE - DO NOT USE")
    print("=" * 70)
    print()
    print("This IPAdapter-based pipeline has been superseded.")
    print()
    print("PROBLEM: IPAdapter generates NEW content influenced by references,")
    print("but cannot replicate exact textures. The armor will look 'similar'")
    print("but NOT match the reference images.")
    print()
    print("SOLUTION: Use the optical flow pipeline instead:")
    print()
    print("    python generate_sprite_clothing_optical.py \\")
    print("        --base input/mannequin_spritesheet.png \\")
    print("        --clothed input/clothed_spritesheet.png \\")
    print("        --output output/result_spritesheet.png")
    print()
    print("Benefits of optical flow:")
    print("  - 100% exact armor preservation (pixel-perfect)")
    print("  - <1s per frame on CPU (vs 10-30s on GPU)")
    print("  - No ComfyUI/GPU dependencies")
    print("  - Deterministic output")
    print()
    print("See: docs/findings/2025-12-13-depth-controlnet-orientation-fix.md")
    print("=" * 70)
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
