"""Neural network model for sprite keypoint detection."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple

from .keypoints import NUM_KEYPOINTS


class SpriteKeypointDetector(nn.Module):
    """Keypoint detector using frozen ResNet18 backbone + trainable head."""

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
        """Initialize the model.

        Args:
            num_keypoints: Number of keypoints to detect
            pretrained: Whether to use pretrained ResNet18 weights
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # Load pretrained ResNet18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Remove the final FC layer and avgpool - keep conv layers
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Custom keypoint regression head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_keypoints * 2),
            nn.Sigmoid()
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize head weights."""
        for m in self.keypoint_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Keypoint coordinates of shape (batch, num_keypoints, 2)
            Values are normalized to [0, 1] range
        """
        with torch.no_grad():
            features = self.backbone(x)

        keypoints = self.keypoint_head(features)
        return keypoints.view(-1, self.num_keypoints, 2)

    def predict(self, x: torch.Tensor, image_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """Predict keypoints in pixel coordinates."""
        normalized = self.forward(x)
        scale = torch.tensor([image_size[0], image_size[1]], device=x.device, dtype=x.dtype)
        return normalized * scale

    def get_trainable_params(self):
        """Get only the trainable parameters (head only)."""
        return self.keypoint_head.parameters()

    def unfreeze_backbone(self, num_layers: int = 1):
        """Unfreeze the last N layers of backbone for fine-tuning."""
        layers = [self.backbone[4], self.backbone[5], self.backbone[6], self.backbone[7]]
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = SpriteKeypointDetector()
    trainable, total = count_parameters(model)

    print(f"Model created:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {total - trainable:,}")

    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
