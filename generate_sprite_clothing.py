#!/usr/bin/env python3
"""Command-line interface for sprite clothing generation.

Usage:
    # Simplest usage (auto-discovers frame_*.png in examples/input/):
    python generate_sprite_clothing.py

    # With custom paths:
    python generate_sprite_clothing.py \
        --base path/to/base_spritesheet.png \
        --reference path/to/frame_012.png \
        --output path/to/output_clothing_spritesheet.png
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from sprite_clothing_gen.orchestrator import SpriteClothingGenerator
from sprite_clothing_gen.config import COMFYUI_URL


def start_comfyui_server():
    """Start ComfyUI server as a subprocess.

    Returns:
        subprocess.Popen: The server process
    """
    print("Starting ComfyUI server...")

    # Start main.py in the current directory
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready (poll for health check)
    from sprite_clothing_gen.comfy_client import ComfyUIClient
    client = ComfyUIClient(COMFYUI_URL)

    max_wait = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if client.health_check():
            print(f"ComfyUI server ready at {COMFYUI_URL}")
            return process
        time.sleep(1)

    # If we get here, server didn't start in time
    process.terminate()
    raise RuntimeError(f"ComfyUI server failed to start within {max_wait} seconds")


def stop_comfyui_server(process):
    """Stop ComfyUI server subprocess.

    Args:
        process: The server subprocess to stop
    """
    if process is None:
        return

    print("\nStopping ComfyUI server...")

    # Try graceful shutdown first
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown fails
        process.kill()
        process.wait()

    print("ComfyUI server stopped")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate clothing-only spritesheet from base spritesheet and reference frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simplest usage (auto-discovers first frame_*.png in examples/input/):
    python generate_sprite_clothing.py

    # Specify which frame file to use:
    python generate_sprite_clothing.py --reference examples/input/frame_005.png

    # Full custom paths:
    python generate_sprite_clothing.py \\
        --base custom/base.png \\
        --reference custom/frame_012.png \\
        --output custom/output.png
        """
    )

    parser.add_argument(
        '--base',
        type=Path,
        default=None,
        help='Path to base 5x5 spritesheet (default: examples/input/base.png)'
    )

    parser.add_argument(
        '--reference',
        type=Path,
        default=None,
        help='Path to single clothed reference frame (default: auto-discovered from examples/input/frame_*.png)'
    )

    parser.add_argument(
        '--frame',
        type=int,
        default=None,
        help='Frame index that reference corresponds to (default: inferred from reference filename)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path to save output clothing spritesheet (default: examples/output/clothing.png)'
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

    # Apply defaults
    comfyui_root = Path(__file__).parent

    if args.base is None:
        args.base = comfyui_root / "examples" / "input" / "base.png"

    # Auto-discover reference frame if not specified
    if args.reference is None:
        input_dir = comfyui_root / "examples" / "input"
        frame_files = sorted(input_dir.glob("frame_*.png"))

        if not frame_files:
            print("Error: No frame_*.png files found in examples/input/", file=sys.stderr)
            print("Please provide a reference frame file or use --reference", file=sys.stderr)
            return 1

        # Use first frame file found (alphabetically)
        args.reference = frame_files[0]
        print(f"Auto-discovered reference: {args.reference.name}")

    # Extract frame number from reference filename if not specified
    if args.frame is None:
        # Extract from filename: frame_012.png → 12
        match = re.search(r'frame_(\d+)', args.reference.name)

        if not match:
            print(f"Error: Cannot infer frame number from filename: {args.reference.name}", file=sys.stderr)
            print("Filename must match pattern 'frame_NNN.png' or use --frame", file=sys.stderr)
            return 1

        # Convert to int (strips leading zeros)
        args.frame = int(match.group(1))
        print(f"Inferred frame index: {args.frame}")

    # Validate frame index
    if not 0 <= args.frame < 25:
        print(f"Error: Frame index must be 0-24, got {args.frame}", file=sys.stderr)
        return 1

    if args.output is None:
        args.output = comfyui_root / "examples" / "output" / "clothing.png"

    # Validate inputs
    if not args.base.exists():
        print(f"Error: Base spritesheet not found: {args.base}", file=sys.stderr)
        return 1

    if not args.reference.exists():
        print(f"Error: Reference frame not found: {args.reference}", file=sys.stderr)
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Setup: Start ComfyUI server
    server_process = None
    try:
        server_process = start_comfyui_server()

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

    finally:
        # Teardown: Stop ComfyUI server
        stop_comfyui_server(server_process)


if __name__ == '__main__':
    sys.exit(main())
