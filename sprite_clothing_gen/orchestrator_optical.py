"""Orchestrator for sprite clothing generation using optical flow."""

import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

from sprite_clothing_gen.optical_flow import warp_clothing_to_pose, create_body_mask, load_image_bgr
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
        debug: bool = False,
        skip_validation: bool = False
    ) -> Path:
        """Generate clothing spritesheet using optical flow warping.

        Args:
            base_spritesheet: Path to base mannequin spritesheet
            clothed_spritesheet: Path to clothed reference spritesheet
            output_path: Where to save output
            grid_size: Grid dimensions (cols, rows)
            keep_temp: Keep temporary files
            debug: Save debug outputs
            skip_validation: Skip interactive mask validation

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
            warped_frames = self._warp_frames(base_frames, clothed_frames, debug)
            print(f"   Warped {len(warped_frames)} frames")

            # Step 4: Generate initial masks (clothing = difference from base)
            print("\nStep 4: Generating clothing masks...")
            masks_dir = self.temp_dir / "masks"
            self._generate_masks(base_frames, warped_frames, masks_dir)

            # Step 5: Interactive mask validation
            if not skip_validation:
                print("\nStep 5: Interactive mask validation...")
                self._validate_masks(base_frames, warped_frames, masks_dir)
            else:
                print("\nStep 5: Skipping mask validation (--skip-validation)")

            # Step 6: Extract clothing-only spritesheet
            print("\nStep 6: Extracting clothing spritesheet...")
            clothing_frames = self._extract_clothing(warped_frames, masks_dir)

            # Step 7: Reassemble
            print("\nStep 7: Reassembling spritesheet...")
            result = reassemble_spritesheet(clothing_frames, output_path, grid_size)
            print(f"   Saved to {output_path}")

            # Create verification overlay
            print("\nStep 8: Creating verification overlay...")
            self._create_verification(base_spritesheet, output_path)

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
        warped_dir = self.temp_dir / "warped"
        warped_dir.mkdir(exist_ok=True)
        debug_dir = self.temp_dir / "debug" if debug else None

        warped = []
        skipped_count = 0
        for i, (base, clothed) in enumerate(zip(base_frames, clothed_frames)):
            output = warped_dir / f"warped_{i:02d}.png"
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

    def _generate_masks(
        self,
        base_frames: List[Path],
        warped_frames: List[Path],
        masks_dir: Path
    ) -> List[Path]:
        """Generate clothing masks using trained U-Net model.

        Uses the pre-trained clothing segmentation model for accurate masks.
        Falls back to pixel difference if model not available.
        """
        masks_dir.mkdir(exist_ok=True)
        masks = []

        # Try to use trained U-Net model
        model_path = Path(__file__).parent.parent / "models" / "clothing_segmentation_unet.pth"
        if model_path.exists():
            masks = self._generate_masks_with_model(warped_frames, masks_dir, model_path)
        else:
            print(f"   Warning: Trained model not found at {model_path}")
            print(f"   Falling back to pixel difference method (less accurate)")
            masks = self._generate_masks_pixel_diff(base_frames, warped_frames, masks_dir)

        return masks

    def _generate_masks_with_model(
        self,
        warped_frames: List[Path],
        masks_dir: Path,
        model_path: Path
    ) -> List[Path]:
        """Generate masks using trained U-Net model."""
        import torch
        import torch.nn as nn
        import torchvision.transforms.functional as TF

        # Import model architecture
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from predict_masks_with_model import LightweightUNet

        # Setup device
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"   Using device: {device}")

        # Load model
        model = LightweightUNet().to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"   Loaded trained U-Net model")

        masks = []
        for i, warped_path in enumerate(warped_frames):
            # Load image
            warped = Image.open(warped_path).convert('RGB')
            warped_tensor = TF.to_tensor(warped).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(warped_tensor)

            # Convert to binary mask
            mask_arr = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

            # Save mask
            mask_path = masks_dir / f"mask_{i:02d}.png"
            Image.fromarray(mask_arr).save(mask_path)
            masks.append(mask_path)

            clothing_pixels = np.sum(mask_arr > 128)
            print(f"   Frame {i+1}: {clothing_pixels:6d} clothing pixels")

        return masks

    def _generate_masks_pixel_diff(
        self,
        base_frames: List[Path],
        warped_frames: List[Path],
        masks_dir: Path
    ) -> List[Path]:
        """Fallback: Generate masks by comparing warped vs base frames.

        Less accurate than trained model - marks any differing pixel as clothing.
        """
        masks = []

        for i, (base_path, warped_path) in enumerate(zip(base_frames, warped_frames)):
            # Load images
            base = np.array(Image.open(base_path).convert('RGB'))
            warped = np.array(Image.open(warped_path).convert('RGB'))

            # Find pixels that differ between base and warped
            diff = np.abs(base.astype(np.int16) - warped.astype(np.int16))
            diff_sum = diff.sum(axis=2)

            # Threshold: pixels with significant color difference are clothing
            threshold = 30
            clothing_mask = (diff_sum > threshold).astype(np.uint8) * 255

            # Also exclude pure white background
            warped_gray = np.mean(warped, axis=2)
            not_white = (warped_gray < 250).astype(np.uint8) * 255
            clothing_mask = np.minimum(clothing_mask, not_white)

            # Save mask
            mask_path = masks_dir / f"mask_{i:02d}.png"
            Image.fromarray(clothing_mask).save(mask_path)
            masks.append(mask_path)

            clothing_pixels = np.sum(clothing_mask > 128)
            print(f"   Frame {i+1}: {clothing_pixels:6d} clothing pixels")

        return masks

    def _validate_masks(
        self,
        base_frames: List[Path],
        warped_frames: List[Path],
        masks_dir: Path
    ):
        """Interactive mask validation using matplotlib GUI.

        Loads base frames as RGBA to detect transparent pixels.
        After each edit, applies remove_transparent_background to clean mask.
        """
        # Import here to avoid requiring matplotlib for headless operation
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Ensure interactive backend
        except:
            pass

        # Add parent directory to path for mask_correction_tool import
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from mask_correction_tool import MaskEditor, remove_transparent_background
        import matplotlib.pyplot as plt

        print("=" * 70)
        print("MASK VALIDATION")
        print("=" * 70)
        print()
        print("Controls:")
        print("  Left Click: Add clothing pixels (paint red)")
        print("  Right Click: Remove clothing pixels (erase)")
        print("  Scroll: Adjust brush size")
        print("  Ctrl + Scroll: Zoom in/out")
        print("  Cmd+Z / Ctrl+Z: Undo")
        print("  Save Button: Accept and continue to next frame")
        print("  Cancel Button: Skip this frame")
        print()
        print("Note: Transparent background pixels are automatically excluded")
        print("=" * 70)

        for i, (base_path, warped_path) in enumerate(zip(base_frames, warped_frames)):
            mask_path = masks_dir / f"mask_{i:02d}.png"

            if not mask_path.exists():
                print(f"   Frame {i}: No mask found, skipping")
                continue

            print(f"\n   Reviewing frame {i+1}/{len(base_frames)}...")

            # Load base as RGBA to get alpha channel
            base_rgba = np.array(Image.open(base_path).convert('RGBA'))
            base_rgb = base_rgba[:, :, :3]

            # Load warped (clothed) image
            warped_img = np.array(Image.open(warped_path).convert('RGB'))

            # Load existing mask
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 128).astype(np.uint8)

            # Launch editor
            editor = MaskEditor(
                base_img=base_rgb,
                clothed_img=warped_img,
                mask=mask
            )
            plt.show()

            # Clean mask: remove any clothing labels on transparent pixels
            corrected_mask = remove_transparent_background(editor.mask, base_rgba)

            # Save corrected mask
            mask_output = (corrected_mask * 255).astype(np.uint8)
            Image.fromarray(mask_output).save(mask_path)

            cleaned_pixels = (editor.mask.sum() - corrected_mask.sum())
            if cleaned_pixels > 0:
                print(f"   → Saved mask {i+1} (removed {cleaned_pixels} transparent bg pixels)")
            else:
                print(f"   → Saved mask {i+1}")

        print()
        print("=" * 70)
        print("✓ Mask validation complete!")
        print("=" * 70)

    def _extract_clothing(
        self,
        warped_frames: List[Path],
        masks_dir: Path
    ) -> List[Path]:
        """Extract clothing-only frames using validated masks."""
        clothing_dir = self.temp_dir / "clothing"
        clothing_dir.mkdir(exist_ok=True)

        clothing_frames = []
        for i, warped_path in enumerate(warped_frames):
            mask_path = masks_dir / f"mask_{i:02d}.png"

            # Load warped frame and mask
            warped = np.array(Image.open(warped_path).convert('RGBA'))
            mask = np.array(Image.open(mask_path).convert('L'))

            # Apply mask to alpha channel
            clothing = warped.copy()
            clothing[:, :, 3] = np.where(mask > 128, 255, 0)

            # Save clothing-only frame
            clothing_path = clothing_dir / f"clothing_{i:02d}.png"
            Image.fromarray(clothing).save(clothing_path)
            clothing_frames.append(clothing_path)

            clothing_pixels = np.sum(mask > 128)
            print(f"   Frame {i+1}: {clothing_pixels:6d} clothing pixels extracted")

        return clothing_frames

    def _create_verification(self, base_spritesheet: Path, clothing_spritesheet: Path):
        """Create overlay of clothing on base for visual verification."""
        base = Image.open(base_spritesheet).convert('RGBA')
        clothing = Image.open(clothing_spritesheet).convert('RGBA')

        # Composite clothing over base
        overlay = Image.alpha_composite(base, clothing)

        # Save verification image
        verification_path = clothing_spritesheet.parent / f"{clothing_spritesheet.stem}_verification.png"
        overlay.save(verification_path)
        print(f"   Verification overlay: {verification_path}")

    def _cleanup(self):
        """Remove temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
