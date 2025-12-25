"""U-Net mask prediction model for clothing segmentation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetSmall(nn.Module):
    """Small U-Net for mask prediction from (base, clothed) image pairs."""

    def __init__(self):
        super().__init__()
        # Input: 8 channels (base RGBA + clothed RGBA)
        # Encoder
        self.enc1 = DoubleConv(8, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # Output
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))


class MaskDataset(Dataset):
    """Dataset for mask prediction training."""

    def __init__(self, samples, augment=True):
        """
        samples: list of (base_path, clothed_path, mask_path) tuples
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_path, clothed_path, mask_path = self.samples[idx]

        # Load images
        base = cv2.imread(str(base_path), cv2.IMREAD_UNCHANGED)
        clothed = cv2.imread(str(clothed_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize to 256x256 for training efficiency
        base = cv2.resize(base, (256, 256), interpolation=cv2.INTER_AREA)
        clothed = cv2.resize(clothed, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Data augmentation
        if self.augment and random.random() > 0.5:
            base = cv2.flip(base, 1)
            clothed = cv2.flip(clothed, 1)
            mask = cv2.flip(mask, 1)

        # Normalize to [0, 1]
        base = base.astype(np.float32) / 255.0
        clothed = clothed.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # Stack base and clothed as input (8 channels: RGBA + RGBA)
        x = np.concatenate([base, clothed], axis=2)  # H, W, 8
        x = np.transpose(x, (2, 0, 1))  # 8, H, W

        # Mask as output
        y = mask[np.newaxis, :, :]  # 1, H, W

        return torch.FloatTensor(x), torch.FloatTensor(y)


def gather_training_data(training_dir):
    """Gather all training samples from walk_north and walk_south backups."""
    training_dir = Path(training_dir)
    samples = []

    # Walk north data
    north_frames = training_dir / 'frames_backup_walk_north'
    north_masks = training_dir / 'masks_initial_backup_walk_north'

    if north_frames.exists() and north_masks.exists():
        for i in range(25):
            base = north_frames / f'base_frame_{i:02d}.png'
            clothed = north_frames / f'clothed_frame_{i:02d}.png'
            mask = north_masks / f'mask_{i:02d}.png'
            if base.exists() and clothed.exists() and mask.exists():
                samples.append((base, clothed, mask))

    # Walk south data (from backup)
    south_frames = training_dir / 'frames_backup_walk_south'
    south_masks = training_dir / 'masks_backup_walk_south'

    if south_frames.exists() and south_masks.exists():
        for i in range(25):
            base = south_frames / f'base_frame_{i:02d}.png'
            clothed = south_frames / f'clothed_frame_{i:02d}.png'
            mask = south_masks / f'mask_{i:02d}.png'
            if base.exists() and clothed.exists() and mask.exists():
                samples.append((base, clothed, mask))

    return samples


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for better segmentation with class imbalance."""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # BCE loss
        bce_loss = self.bce(pred, target)

        # Dice loss
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_model(training_dir, epochs=100, lr=1e-3, save_path='mask_model.pth'):
    """Train the mask prediction model."""
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Gather data
    samples = gather_training_data(training_dir)
    print(f'Found {len(samples)} training samples')

    if len(samples) == 0:
        raise ValueError('No training samples found!')

    # Create dataset and dataloader
    dataset = MaskDataset(samples, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create model
    model = UNetSmall().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss(bce_weight=0.3)  # Weight towards Dice for class imbalance

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    return model


def predict_mask(model, base_path, clothed_path, device=None):
    """Predict mask for a single image pair."""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess
    base = cv2.imread(str(base_path), cv2.IMREAD_UNCHANGED)
    clothed = cv2.imread(str(clothed_path), cv2.IMREAD_UNCHANGED)

    orig_h, orig_w = base.shape[:2]

    base_resized = cv2.resize(base, (256, 256), interpolation=cv2.INTER_AREA)
    clothed_resized = cv2.resize(clothed, (256, 256), interpolation=cv2.INTER_AREA)

    base_norm = base_resized.astype(np.float32) / 255.0
    clothed_norm = clothed_resized.astype(np.float32) / 255.0

    x = np.concatenate([base_norm, clothed_norm], axis=2)
    x = np.transpose(x, (2, 0, 1))
    x = torch.FloatTensor(x).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(x)

    # Post-process
    mask = pred[0, 0].cpu().numpy()
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 0.5).astype(np.uint8) * 255

    return mask


def load_model(model_path, device=None):
    """Load a trained model."""
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetSmall().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


if __name__ == '__main__':
    import sys

    training_dir = sys.argv[1] if len(sys.argv) > 1 else 'training_data'
    model = train_model(training_dir, epochs=100)
