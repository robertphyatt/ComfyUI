#!/usr/bin/env python3
"""Use trained U-Net model to predict masks for complete frames."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
import argparse
import sys


class LightweightUNet(nn.Module):
    """Lightweight U-Net for binary segmentation."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, 1, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))


def predict_mask(model, frame_path: Path, device):
    """Predict clothing mask for a single frame."""
    # Load image
    base = Image.open(frame_path.parent.parent / "frames" / frame_path.name).convert('RGB')
    clothed = Image.open(frame_path).convert('RGB')

    # Convert to tensors
    base_tensor = TF.to_tensor(base).unsqueeze(0).to(device)
    clothed_tensor = TF.to_tensor(clothed).unsqueeze(0).to(device)

    # Create input (clothed RGB)
    input_tensor = clothed_tensor

    # Predict
    with torch.no_grad():
        output = model(input_tensor)

    # Convert to binary mask
    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    return Image.fromarray(mask)


def main():
    """Generate masks for all complete frames using trained model."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Predict masks using trained U-Net model. "
                    "SAFETY: Will not overwrite existing masks without --force."
    )
    parser.add_argument('--frames-dir', type=Path,
                        default=Path('training_data/frames_complete'),
                        help="Directory containing clothed frames")
    parser.add_argument('--output-dir', type=Path,
                        default=Path('training_data/masks_predicted'),
                        help="Output directory for predicted masks (default: masks_predicted/)")
    parser.add_argument('--animation', type=str, default=None,
                        help="Animation name - outputs to training_data/animations/<name>/masks_predicted/")
    parser.add_argument('--force', action='store_true',
                        help="Force overwrite even if masks exist or .verified marker present")
    args = parser.parse_args()

    # Resolve paths based on --animation flag
    if args.animation:
        anim_dir = Path("training_data/animations") / args.animation
        if not anim_dir.exists():
            parser.error(f"Animation directory not found: {anim_dir}")
        complete_dir = anim_dir / "frames"
        # Output to staging area, NOT the canonical masks/ directory
        output_dir = anim_dir / "masks_predicted"
        canonical_masks_dir = anim_dir / "masks"
    else:
        complete_dir = args.frames_dir
        output_dir = args.output_dir
        canonical_masks_dir = None

    # Safety check: refuse to overwrite canonical masks without --force
    if canonical_masks_dir and output_dir == canonical_masks_dir:
        print("ERROR: Cannot write directly to canonical masks/ directory.")
        print("       Predictions go to masks_predicted/ for review first.")
        print("       Use --output-dir to specify a different location.")
        sys.exit(1)

    # Safety check: look for .verified marker
    verified_marker = output_dir / ".verified"
    if verified_marker.exists() and not args.force:
        print("="*70)
        print("SAFETY STOP: Found .verified marker in output directory")
        print(f"  {verified_marker}")
        print()
        print("This means masks were manually verified. Refusing to overwrite.")
        print("Use --force to override (will delete .verified marker)")
        print("="*70)
        sys.exit(1)

    # Safety check: existing masks
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_masks = list(output_dir.glob("mask_*.png"))
    if existing_masks and not args.force:
        print("="*70)
        print(f"SAFETY STOP: {len(existing_masks)} existing masks found in {output_dir}")
        print()
        print("Refusing to overwrite without --force flag.")
        print("If you want to regenerate, use: --force")
        print("="*70)
        sys.exit(1)

    # If --force and .verified exists, remove the marker
    if args.force and verified_marker.exists():
        verified_marker.unlink()
        print(f"Removed .verified marker: {verified_marker}")

    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = Path("models/clothing_segmentation_unet.pth")

    # Load model
    print("=" * 70)
    print("PREDICTING MASKS WITH TRAINED U-NET MODEL")
    print("=" * 70)
    print()
    print(f"Frames: {complete_dir}")
    print(f"Output: {output_dir}")
    if args.force:
        print("Mode: FORCE (overwriting existing)")
    print()
    print(f"Loading model from {model_path}...")

    model = LightweightUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully on {device}")
    print()

    # Predict masks
    for frame_idx in range(25):
        frame_path = complete_dir / f"clothed_frame_{frame_idx:02d}.png"

        if not frame_path.exists():
            print(f"Frame {frame_idx:02d}: SKIPPED (not found)")
            continue

        # Predict mask
        mask = predict_mask(model, frame_path, device)

        # Count pixels
        mask_arr = np.array(mask)
        pixels = np.sum(mask_arr > 128)

        # Save
        output_path = output_dir / f"mask_{frame_idx:02d}.png"
        mask.save(output_path)

        print(f"Frame {frame_idx:02d}: {pixels:6d} pixels predicted -> {output_path}")

    print()
    print("=" * 70)
    print(f"Predicted masks saved to {output_dir}/")
    print()
    print("NEXT STEPS:")
    print(f"  1. Review predictions in {output_dir}/")
    print(f"  2. Run mask_correction_tool.py to fix any issues")
    if canonical_masks_dir:
        print(f"  3. Copy verified masks to canonical location:")
        print(f"     cp {output_dir}/mask_*.png {canonical_masks_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
