#!/usr/bin/env python3
"""Command-line interface for sprite clothing generation.

Usage:
    python generate_sprite_clothing.py \
        --base path/to/base_spritesheet.png \
        --reference path/to/clothed_frame.png \
        --frame 12 \
        --output path/to/output_clothing_spritesheet.png \
        --seed 42 \
        --keep-temp
"""

import argparse
import sys
from pathlib import Path
from sprite_clothing_gen.orchestrator import SpriteClothingGenerator
from sprite_clothing_gen.config import COMFYUI_URL


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate clothing-only spritesheet from base spritesheet and reference frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python generate_sprite_clothing.py \
        --base input/base_spritesheet.png \
        --reference input/clothed_frame.png \
        --frame 12 \
        --output output/clothing_spritesheet.png
        """
    )

    parser.add_argument(
        '--base',
        type=Path,
        required=True,
        help='Path to base 5x5 spritesheet (25 frames, naked character)'
    )

    parser.add_argument(
        '--reference',
        type=Path,
        required=True,
        help='Path to single clothed reference frame from Ludo'
    )

    parser.add_argument(
        '--frame',
        type=int,
        required=True,
        help='Frame index that reference corresponds to (0-24, zero-indexed)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output/clothing_spritesheet.png'),
        help='Path to save output clothing spritesheet (default: output/clothing_spritesheet.png)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for generation (default: 42)'
    )

    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary files for debugging'
    )

    parser.add_argument(
        '--comfyui-url',
        type=str,
        default=COMFYUI_URL,
        help=f'ComfyUI server URL (default: {COMFYUI_URL})'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.base.exists():
        print(f"Error: Base spritesheet not found: {args.base}", file=sys.stderr)
        return 1

    if not args.reference.exists():
        print(f"Error: Reference frame not found: {args.reference}", file=sys.stderr)
        return 1

    if not 0 <= args.frame < 25:
        print(f"Error: Frame index must be 0-24, got {args.frame}", file=sys.stderr)
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize generator
        generator = SpriteClothingGenerator(args.comfyui_url)

        # Run generation
        result = generator.generate(
            base_spritesheet=args.base,
            reference_frame=args.reference,
            reference_frame_index=args.frame,
            output_path=args.output,
            seed=args.seed,
            keep_temp=args.keep_temp
        )

        print(f"\nSuccess! Clothing spritesheet saved to: {result}")
        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
