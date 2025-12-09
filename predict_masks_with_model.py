#!/usr/bin/env python3
"""Use trained U-Net model to predict masks for complete frames."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-dir', type=Path, default=Path('training_data/frames_complete'))
    parser.add_argument('--output-dir', type=Path, default=Path('training_data/masks_predicted'))
    args = parser.parse_args()

    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = Path("models/clothing_segmentation_unet.pth")
    complete_dir = args.frames_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("=" * 70)
    print("PREDICTING MASKS WITH TRAINED U-NET MODEL")
    print("=" * 70)
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
    print(f"✓ All predicted masks saved to {output_dir}/")
    print("=" * 70)
    print()
    print("Review masks and compare to your manual corrections")


if __name__ == "__main__":
    main()
