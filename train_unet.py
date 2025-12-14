#!/usr/bin/env python3
"""Train U-Net model on labeled frames and predict remaining masks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
import numpy as np
import random


class LightweightUNet(nn.Module):
    """Lightweight U-Net for binary segmentation."""

    def __init__(self):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # Output
        self.out = nn.Conv2d(32, 1, 1)

        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return torch.sigmoid(self.out(dec1))


class ClothingDataset(Dataset):
    """Dataset for clothing segmentation with data augmentation."""

    def __init__(self, frame_nums, frames_dir, masks_dir, augment=True):
        self.frame_nums = frame_nums
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(masks_dir)
        self.augment = augment

    def __len__(self):
        return len(self.frame_nums) * (4 if self.augment else 1)

    def __getitem__(self, idx):
        # Get base frame index
        frame_idx = idx % len(self.frame_nums)
        aug_idx = idx // len(self.frame_nums)
        frame_num = self.frame_nums[frame_idx]

        # Load images
        clothed_path = self.frames_dir / f"clothed_frame_{frame_num:02d}.png"
        mask_path = self.masks_dir / f"mask_{frame_num:02d}.png"

        clothed = Image.open(clothed_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Apply augmentation based on aug_idx
        if self.augment and aug_idx > 0:
            if aug_idx == 1:  # Horizontal flip
                clothed = TF.hflip(clothed)
                mask = TF.hflip(mask)
            elif aug_idx == 2:  # Vertical flip
                clothed = TF.vflip(clothed)
                mask = TF.vflip(mask)
            elif aug_idx == 3:  # Both flips
                clothed = TF.hflip(TF.vflip(clothed))
                mask = TF.hflip(TF.vflip(mask))

        # Convert to tensors
        clothed_tensor = TF.to_tensor(clothed)
        mask_tensor = TF.to_tensor(mask)

        return clothed_tensor, mask_tensor


def train_model(model, train_loader, device, epochs=100, lr=0.001):
    """Train the U-Net model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def predict_masks(model, frame_nums, frames_dir, output_dir, device):
    """Predict masks for given frames."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for frame_num in frame_nums:
            clothed_path = Path(frames_dir) / f"clothed_frame_{frame_num:02d}.png"

            # Load and prepare image
            clothed = Image.open(clothed_path).convert('RGB')
            image_tensor = TF.to_tensor(clothed).unsqueeze(0).to(device)

            # Predict
            output = model(image_tensor)

            # Convert to binary mask
            mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            # Save mask
            mask_img = Image.fromarray(mask * 255)
            mask_path = output_dir / f"mask_{frame_num:02d}.png"
            mask_img.save(mask_path)

            # Statistics
            clothing_pixels = np.sum(mask == 1)
            percent = 100 * clothing_pixels / (512 * 512)
            print(f"Frame {frame_num:02d}: Predicted {clothing_pixels} pixels ({percent:.1f}%)")


def main():
    # Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    frames_dir = Path("training_data/frames")
    corrected_masks_dir = Path("training_data/masks_corrected")
    initial_masks_dir = Path("training_data/masks_initial")

    # Training frames (0-17 that you've already labeled)
    train_frames = list(range(18))
    # Prediction frames (18-24 that need better initial masks)
    predict_frames = list(range(18, 25))

    print("=" * 70)
    print("Training U-Net on corrected masks")
    print("=" * 70)
    print(f"Training frames: {len(train_frames)}")
    print(f"Augmentation: 4x (original + 3 flips)")
    print(f"Total training samples: {len(train_frames) * 4}")
    print()

    # Create dataset and dataloader
    train_dataset = ClothingDataset(train_frames, frames_dir, corrected_masks_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Create and train model
    model = LightweightUNet().to(device)
    print("Training...")
    model = train_model(model, train_loader, device, epochs=100, lr=0.001)

    print()
    print("=" * 70)
    print("Generating predictions for remaining frames")
    print("=" * 70)
    print()

    # Predict remaining frames
    predict_masks(model, predict_frames, frames_dir, initial_masks_dir, device)

    print()
    print("=" * 70)
    print(f"✓ Model-based masks saved to {initial_masks_dir}/")
    print()
    print("Next: Continue manual corrections with improved initial masks!")
    print("Run: python3 mask_correction_tool.py")


if __name__ == "__main__":
    main()
