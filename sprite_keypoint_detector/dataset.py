"""Dataset class for sprite keypoint training."""

import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


class SpriteKeypointDataset(Dataset):
    """Dataset for sprite keypoint detection training."""

    def __init__(
        self,
        annotations_json: Path,
        image_dir: Path,
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

        with open(annotations_json) as f:
            all_annotations = json.load(f)

        self.annotations = []
        for img_name, data in all_annotations.items():
            keypoints = data.get("keypoints", {})
            if len(keypoints) == NUM_KEYPOINTS:
                self.annotations.append({
                    "image": img_name,
                    "keypoints": keypoints
                })

        print(f"Loaded {len(self.annotations)} fully annotated images")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ann = self.annotations[idx]

        img_path = self.image_dir / ann["image"]
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size

        keypoints = []
        for name in KEYPOINT_NAMES:
            x, y = ann["keypoints"][name]
            keypoints.append([x / orig_size[0], y / orig_size[1]])

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        if self.augment:
            image, keypoints = self._augment(image, keypoints)

        image_tensor = self.transform(image)
        return image_tensor, keypoints

    def _augment(self, image: Image.Image, keypoints: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            keypoints[:, 0] = 1.0 - keypoints[:, 0]
            swap_pairs = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
            for i, j in swap_pairs:
                keypoints[i], keypoints[j] = keypoints[j].clone(), keypoints[i].clone()
        return image, keypoints


def create_data_loaders(
    annotations_json: Path,
    image_dir: Path,
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    full_dataset = SpriteKeypointDataset(annotations_json, image_dir, augment=False)

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    train_dataset.dataset.augment = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader
