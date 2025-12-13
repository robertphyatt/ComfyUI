#!/usr/bin/env python3
"""CLI for sprite clothing generation using optical flow."""

import argparse
from pathlib import Path
from sprite_clothing_gen.orchestrator_optical import SpriteClothingGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate clothing spritesheet using optical flow warping"
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Path to base mannequin spritesheet"
    )
    parser.add_argument(
        "--clothed",
        type=Path,
        required=True,
        help="Path to clothed reference spritesheet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output clothing spritesheet"
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="5x5",
        help="Grid size as WxH (default: 5x5)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug outputs"
    )

    args = parser.parse_args()

    # Parse grid size
    w, h = map(int, args.grid.lower().split('x'))
    grid_size = (w, h)

    # Validate inputs
    if not args.base.exists():
        raise FileNotFoundError(f"Base not found: {args.base}")
    if not args.clothed.exists():
        raise FileNotFoundError(f"Clothed not found: {args.clothed}")

    # Run
    generator = SpriteClothingGenerator()
    generator.generate(
        base_spritesheet=args.base,
        clothed_spritesheet=args.clothed,
        output_path=args.output,
        grid_size=grid_size,
        keep_temp=args.keep_temp,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
