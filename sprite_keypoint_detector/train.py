"""Training script for sprite keypoint detector."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import numpy as np

from .model import SpriteKeypointDetector, count_parameters
from .dataset import create_data_loaders
from .keypoints import NUM_KEYPOINTS


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, keypoints in loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, keypoints)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device, image_size: Tuple[int, int] = (512, 512)) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, keypoints in loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            predictions = model(images)
            loss = criterion(predictions, keypoints)
            total_loss += loss.item()

            # Scale x and y coordinates separately for non-square images
            scale = torch.tensor([image_size[0], image_size[1]], device=device, dtype=predictions.dtype)
            pred_px = predictions * scale
            gt_px = keypoints * scale
            error = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1)).mean()
            total_error += error.item() * images.size(0)
            n_samples += images.size(0)

    return {
        'loss': total_loss / len(loader),
        'mean_error_px': total_error / n_samples
    }


def train(
    annotations_json: Path,
    image_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: Optional[str] = None
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    train_loader, val_loader = create_data_loaders(annotations_json, image_dir, batch_size=batch_size)

    model = SpriteKeypointDetector(num_keypoints=NUM_KEYPOINTS)
    model = model.to(device)

    trainable, total = count_parameters(model)
    print(f"Model: {trainable:,} trainable / {total:,} total parameters")

    criterion = nn.MSELoss()
    optimizer = AdamW(model.get_trainable_params(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_model_path = output_dir / 'best_model.pth'
    history = {'train_loss': [], 'val_loss': [], 'val_error_px': []}

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_error_px'].append(val_metrics['mean_error_px'])

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_error_px': val_metrics['mean_error_px'],
            }, best_model_path)
            marker = ' *'
        else:
            marker = ''

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | Error: {val_metrics['mean_error_px']:.1f}px{marker}")

    final_path = output_dir / 'final_model.pth'
    torch.save({'epoch': epochs, 'model_state_dict': model.state_dict()}, final_path)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"Training complete! Best model: {best_model_path}")
    print(f"Best validation error: {history['val_error_px'][np.argmin(history['val_loss'])]:.1f}px")

    return best_model_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m sprite_keypoint_detector.train <annotations.json> <image_dir> [output_dir]")
        sys.exit(1)
    train(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else Path('models/sprite_keypoint'))
