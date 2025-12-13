"""Orchestrator for sprite clothing generation using optical flow."""

from pathlib import Path
from typing import List, Tuple
from sprite_clothing_gen.optical_flow import warp_clothing_to_pose
from sprite_clothing_gen.spritesheet_utils import split_spritesheet, reassemble_spritesheet


class SpriteClothingGenerator:
    """Generates clothing spritesheets using optical flow warping."""

    def __init__(self, temp_dir: Path = None):
        """Initialize generator.

        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir or Path("/tmp/sprite_clothing")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        base_spritesheet: Path,
        clothed_spritesheet: Path,
        output_path: Path,
        grid_size: Tuple[int, int] = (5, 5),
        keep_temp: bool = False,
        debug: bool = False
    ) -> Path:
        """Generate clothing spritesheet using optical flow warping.

        Args:
            base_spritesheet: Path to base mannequin spritesheet
            clothed_spritesheet: Path to clothed reference spritesheet
            output_path: Where to save output
            grid_size: Grid dimensions (cols, rows)
            keep_temp: Keep temporary files
            debug: Save debug outputs

        Returns:
            Path to output spritesheet
        """
        print(f"Starting sprite clothing generation (optical flow)...")
        print(f"   Base: {base_spritesheet}")
        print(f"   Clothed: {clothed_spritesheet}")

        try:
            # Step 1: Split base spritesheet
            print("\nStep 1: Splitting base spritesheet...")
            base_dir = self.temp_dir / "base_frames"
            base_frames = split_spritesheet(base_spritesheet, base_dir, grid_size)
            print(f"   Split into {len(base_frames)} frames")

            # Step 2: Split clothed spritesheet
            print("\nStep 2: Splitting clothed spritesheet...")
            clothed_dir = self.temp_dir / "clothed_frames"
            clothed_frames = split_spritesheet(clothed_spritesheet, clothed_dir, grid_size)
            print(f"   Split into {len(clothed_frames)} frames")

            # Step 3: Warp each frame
            print("\nStep 3: Warping clothing frames...")
            clothing_frames = self._warp_frames(base_frames, clothed_frames, debug)
            print(f"   Warped {len(clothing_frames)} frames")

            # Step 4: Reassemble
            print("\nStep 4: Reassembling spritesheet...")
            result = reassemble_spritesheet(clothing_frames, output_path, grid_size)
            print(f"   Saved to {output_path}")

            # Cleanup
            if not keep_temp:
                self._cleanup()

            print("\nComplete!")
            return result

        except Exception as e:
            print(f"\nFailed: {e}")
            raise

    def _warp_frames(
        self,
        base_frames: List[Path],
        clothed_frames: List[Path],
        debug: bool
    ) -> List[Path]:
        """Warp clothed frames to match base poses."""
        clothing_dir = self.temp_dir / "clothing"
        clothing_dir.mkdir(exist_ok=True)
        debug_dir = self.temp_dir / "debug" if debug else None

        warped = []
        skipped_count = 0
        for i, (base, clothed) in enumerate(zip(base_frames, clothed_frames)):
            output = clothing_dir / f"clothing_{i:02d}.png"
            _, was_skipped = warp_clothing_to_pose(clothed, base, output, debug_dir)
            warped.append(output)
            if was_skipped:
                skipped_count += 1
                print(f"   Frame {i+1}/{len(base_frames)} (already aligned, copied)")
            else:
                print(f"   Frame {i+1}/{len(base_frames)} (warped)")

        if skipped_count > 0:
            print(f"   {skipped_count} frames were already aligned and copied directly")

        return warped

    def _cleanup(self):
        """Remove temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
