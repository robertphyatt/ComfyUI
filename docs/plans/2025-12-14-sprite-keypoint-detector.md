# Sprite Keypoint Annotation Tool & Custom Detector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a keypoint annotation GUI and train a custom skeleton detector for chibi pixel art sprites that produces consistent skeletons across base mannequin and clothed/armored variants.

**Architecture:** A matplotlib-based annotation tool saves keypoint data as JSON. A PyTorch model uses a frozen ResNet18 backbone with a trainable keypoint regression head. Training on ~50-100 annotated sprite frames produces a detector optimized for our specific sprite style.

**Tech Stack:** Python, PyTorch, torchvision, matplotlib, PIL, JSON

---

## Keypoint Definition

We detect 14 keypoints:

| Index | Name | Description |
|-------|------|-------------|
| 0 | head | Top center of head |
| 1 | neck | Base of neck/top of spine |
| 2 | left_shoulder | Left shoulder joint |
| 3 | right_shoulder | Right shoulder joint |
| 4 | left_elbow | Left elbow joint |
| 5 | right_elbow | Right elbow joint |
| 6 | left_wrist | Left wrist/hand |
| 7 | right_wrist | Right wrist/hand |
| 8 | left_hip | Left hip joint |
| 9 | right_hip | Right hip joint |
| 10 | left_knee | Left knee joint |
| 11 | right_knee | Right knee joint |
| 12 | left_ankle | Left ankle/foot |
| 13 | right_ankle | Right ankle/foot |

**Note:** "Left" and "right" are from the sprite's perspective (anatomical), not the viewer's.

---

### Task 1: Create Project Structure and Keypoint Definitions

**Files:**
- Create: `sprite_keypoint_detector/__init__.py`
- Create: `sprite_keypoint_detector/keypoints.py`

**Step 1: Create the module directory**

```bash
cd /Users/roberthyatt/Code/ComfyUI
mkdir -p sprite_keypoint_detector
```

**Step 2: Create __init__.py**

Create `sprite_keypoint_detector/__init__.py`:

```python
"""Sprite Keypoint Detector - Custom skeleton detection for chibi pixel art."""

__version__ = "0.1.0"
```

**Step 3: Create keypoints.py**

Create `sprite_keypoint_detector/keypoints.py`:

```python
"""Keypoint definitions for sprite skeleton detection."""

KEYPOINT_NAMES = [
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# Skeleton connections for visualization (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1),   # head -> neck
    (1, 2),   # neck -> left_shoulder
    (1, 3),   # neck -> right_shoulder
    (2, 4),   # left_shoulder -> left_elbow
    (3, 5),   # right_shoulder -> right_elbow
    (4, 6),   # left_elbow -> left_wrist
    (5, 7),   # right_elbow -> right_wrist
    (1, 8),   # neck -> left_hip (via spine, simplified)
    (1, 9),   # neck -> right_hip (via spine, simplified)
    (8, 9),   # left_hip -> right_hip
    (8, 10),  # left_hip -> left_knee
    (9, 11),  # right_hip -> right_knee
    (10, 12), # left_knee -> left_ankle
    (11, 13), # right_knee -> right_ankle
]

# Colors for each limb segment (RGB for visualization)
SKELETON_COLORS = [
    (255, 255, 255),  # head-neck: white
    (0, 255, 0),      # neck-left_shoulder: green
    (255, 165, 0),    # neck-right_shoulder: orange
    (0, 255, 0),      # left_shoulder-left_elbow: green
    (255, 165, 0),    # right_shoulder-right_elbow: orange
    (0, 255, 255),    # left_elbow-left_wrist: cyan
    (255, 0, 255),    # right_elbow-right_wrist: magenta
    (0, 255, 0),      # neck-left_hip: green
    (255, 165, 0),    # neck-right_hip: orange
    (255, 0, 0),      # left_hip-right_hip: red
    (0, 255, 0),      # left_hip-left_knee: green
    (255, 165, 0),    # right_hip-right_knee: orange
    (0, 128, 255),    # left_knee-left_ankle: light blue
    (255, 128, 0),    # right_knee-right_ankle: dark orange
]
```

**Step 4: Verify module imports**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS; print(f'{NUM_KEYPOINTS} keypoints defined')"`

Expected: `14 keypoints defined`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/
git commit -m "feat: add sprite keypoint detector module with 14 keypoint definitions"
```

---

### Task 2: Build Keypoint Annotation Tool

**Files:**
- Create: `sprite_keypoint_detector/annotator.py`

**Step 1: Create annotator.py**

Create `sprite_keypoint_detector/annotator.py`:

```python
"""Interactive keypoint annotation tool for sprite frames."""

import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Dict, List, Tuple

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS, SKELETON_CONNECTIONS, SKELETON_COLORS


class KeypointAnnotator:
    """Interactive GUI for annotating keypoints on sprite images."""

    def __init__(self, image_path: Path, existing_keypoints: Optional[Dict] = None):
        """Initialize annotator.

        Args:
            image_path: Path to sprite image
            existing_keypoints: Optional dict of existing keypoint annotations
        """
        self.image_path = Path(image_path)
        self.image = np.array(Image.open(image_path).convert('RGBA'))

        # Initialize keypoints: None means not yet annotated
        self.keypoints: List[Optional[Tuple[int, int]]] = [None] * NUM_KEYPOINTS

        # Load existing keypoints if provided
        if existing_keypoints:
            for i, name in enumerate(KEYPOINT_NAMES):
                if name in existing_keypoints:
                    self.keypoints[i] = tuple(existing_keypoints[name])

        self.current_keypoint_idx = 0
        self.point_artists = []
        self.line_artists = []
        self.saved = False

        self._setup_gui()

    def _setup_gui(self):
        """Set up the matplotlib GUI."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.canvas.manager.set_window_title(f'Annotate: {self.image_path.name}')

        # Display image
        self.ax.imshow(self.image)
        self.ax.set_title(self._get_title())
        self.ax.axis('off')

        # Draw existing keypoints
        self._draw_skeleton()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Add buttons
        ax_save = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_skip = plt.axes([0.81, 0.02, 0.1, 0.04])
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_clear = plt.axes([0.4, 0.02, 0.15, 0.04])

        self.btn_save = Button(ax_save, 'Save')
        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_clear = Button(ax_clear, 'Clear Current')

        self.btn_save.on_clicked(self._on_save)
        self.btn_skip.on_clicked(self._on_skip)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_clear.on_clicked(self._on_clear)

        # Instructions
        self.fig.text(0.5, 0.97,
            'Click to place keypoint | Arrow keys or buttons to navigate | S to save | Q to quit',
            ha='center', va='top', fontsize=9, color='gray')

    def _get_title(self) -> str:
        """Get title showing current keypoint."""
        name = KEYPOINT_NAMES[self.current_keypoint_idx]
        status = "SET" if self.keypoints[self.current_keypoint_idx] else "NOT SET"
        annotated = sum(1 for kp in self.keypoints if kp is not None)
        return f'[{self.current_keypoint_idx + 1}/{NUM_KEYPOINTS}] {name} ({status}) - {annotated}/{NUM_KEYPOINTS} annotated'

    def _draw_skeleton(self):
        """Draw all keypoints and skeleton connections."""
        # Clear existing artists
        for artist in self.point_artists + self.line_artists:
            artist.remove()
        self.point_artists = []
        self.line_artists = []

        # Draw skeleton lines
        for (i, j), color in zip(SKELETON_CONNECTIONS, SKELETON_COLORS):
            if self.keypoints[i] is not None and self.keypoints[j] is not None:
                x = [self.keypoints[i][0], self.keypoints[j][0]]
                y = [self.keypoints[i][1], self.keypoints[j][1]]
                # Normalize RGB
                rgb = (color[0]/255, color[1]/255, color[2]/255)
                line, = self.ax.plot(x, y, '-', color=rgb, linewidth=2, alpha=0.8)
                self.line_artists.append(line)

        # Draw keypoints
        for i, kp in enumerate(self.keypoints):
            if kp is not None:
                color = 'lime' if i == self.current_keypoint_idx else 'cyan'
                size = 120 if i == self.current_keypoint_idx else 60
                point = self.ax.scatter(kp[0], kp[1], c=color, s=size,
                                        marker='o', edgecolors='white', linewidths=1, zorder=10)
                self.point_artists.append(point)

                # Add label for current keypoint
                if i == self.current_keypoint_idx:
                    text = self.ax.text(kp[0] + 10, kp[1] - 10, KEYPOINT_NAMES[i],
                                       fontsize=8, color='white',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    self.point_artists.append(text)

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click to place keypoint."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # Left click only
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.keypoints[self.current_keypoint_idx] = (x, y)

        # Auto-advance to next unset keypoint
        self._advance_to_next_unset()

        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _advance_to_next_unset(self):
        """Advance to the next unset keypoint, or stay if all are set."""
        start = self.current_keypoint_idx
        for _ in range(NUM_KEYPOINTS):
            self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
            if self.keypoints[self.current_keypoint_idx] is None:
                return
        # All set, stay at current + 1 (or wrap)
        self.current_keypoint_idx = (start + 1) % NUM_KEYPOINTS

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key == 'right' or event.key == 'down':
            self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
        elif event.key == 'left' or event.key == 'up':
            self.current_keypoint_idx = (self.current_keypoint_idx - 1) % NUM_KEYPOINTS
        elif event.key == 's':
            self._on_save(None)
            return
        elif event.key == 'q':
            plt.close(self.fig)
            return
        elif event.key == 'c':
            self._on_clear(None)
            return

        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _on_save(self, event):
        """Save and close."""
        self.saved = True
        plt.close(self.fig)

    def _on_skip(self, event):
        """Skip without saving."""
        self.saved = False
        plt.close(self.fig)

    def _on_prev(self, event):
        """Go to previous keypoint."""
        self.current_keypoint_idx = (self.current_keypoint_idx - 1) % NUM_KEYPOINTS
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _on_next(self, event):
        """Go to next keypoint."""
        self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _on_clear(self, event):
        """Clear current keypoint."""
        self.keypoints[self.current_keypoint_idx] = None
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def run(self) -> Optional[Dict[str, List[int]]]:
        """Run the annotator and return keypoints if saved.

        Returns:
            Dict mapping keypoint names to [x, y] coordinates, or None if skipped
        """
        plt.show()

        if not self.saved:
            return None

        # Convert to dict format
        result = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            if self.keypoints[i] is not None:
                result[name] = list(self.keypoints[i])

        return result


def load_seed_annotations(seed_json: Path) -> Dict:
    """Load seed annotations and convert to annotation tool format.

    Args:
        seed_json: Path to seed_annotations.json (AI-generated initial keypoints)

    Returns:
        Dict in annotation tool format
    """
    with open(seed_json) as f:
        seed_data = json.load(f)

    # Convert from seed format to annotation format
    annotations = {}
    for img_name, data in seed_data.get("annotations", {}).items():
        keypoints = data.get("keypoints", {})
        annotations[img_name] = {
            "image": img_name,
            "keypoints": keypoints
        }

    return annotations


def annotate_directory(
    image_dir: Path,
    output_json: Path,
    pattern: str = "*.png",
    seed_json: Optional[Path] = None
) -> None:
    """Annotate all images in a directory.

    Args:
        image_dir: Directory containing images
        output_json: Path to save/load annotations JSON
        pattern: Glob pattern for image files
        seed_json: Optional path to seed annotations (AI-generated initial keypoints)
    """
    image_dir = Path(image_dir)
    output_json = Path(output_json)

    # Load existing annotations
    annotations = {}
    if output_json.exists():
        with open(output_json) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")
    elif seed_json and Path(seed_json).exists():
        # Load from seed annotations if no existing annotations
        annotations = load_seed_annotations(seed_json)
        print(f"Loaded {len(annotations)} seed annotations for review")

    # Get all images
    images = sorted(image_dir.glob(pattern))
    print(f"Found {len(images)} images")

    # Annotate each
    for i, img_path in enumerate(images):
        img_name = img_path.name
        existing = annotations.get(img_name, {}).get("keypoints")

        status = "DONE" if existing and len(existing) == NUM_KEYPOINTS else "TODO"
        print(f"\n[{i+1}/{len(images)}] {img_name} ({status})")

        if status == "DONE":
            resp = input("  Already annotated. Re-annotate? (y/N): ").strip().lower()
            if resp != 'y':
                continue

        annotator = KeypointAnnotator(img_path, existing)
        result = annotator.run()

        if result:
            annotations[img_name] = {
                "image": img_name,
                "keypoints": result
            }
            # Save after each annotation
            with open(output_json, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"  Saved {len(result)} keypoints")
        else:
            print("  Skipped")

    print(f"\nDone! Total annotations: {len(annotations)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sprite_keypoint_detector.annotator <image_dir> [output.json] [--seed seed.json]")
        print("Example: python -m sprite_keypoint_detector.annotator training_data/frames annotations.json --seed seed_annotations.json")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else image_dir / "annotations.json"

    seed_json = None
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        if idx + 1 < len(sys.argv):
            seed_json = Path(sys.argv[idx + 1])

    annotate_directory(image_dir, output_json, seed_json=seed_json)
```

**Step 2: Test annotator imports**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.annotator import KeypointAnnotator; print('Annotator imported successfully')"`

Expected: `Annotator imported successfully`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/annotator.py
git commit -m "feat: add interactive keypoint annotation tool with seed annotation support"
```

---

### Task 3: Build the Neural Network Model

**Files:**
- Create: `sprite_keypoint_detector/model.py`

**Step 1: Create model.py**

Create `sprite_keypoint_detector/model.py`:

```python
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
```

**Step 2: Test model**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.model`

Expected: Model info and forward pass test output

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/model.py
git commit -m "feat: add ResNet18-based sprite keypoint detector model"
```

---

### Task 4: Build Dataset and Training Pipeline

**Files:**
- Create: `sprite_keypoint_detector/dataset.py`
- Create: `sprite_keypoint_detector/train.py`

**Step 1: Create dataset.py**

Create `sprite_keypoint_detector/dataset.py`:

```python
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
```

**Step 2: Create train.py**

Create `sprite_keypoint_detector/train.py`:

```python
"""Training script for sprite keypoint detector."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from typing import Dict, Optional
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


def validate(model, loader, criterion, device, image_size: int = 512) -> Dict[str, float]:
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

            pred_px = predictions * image_size
            gt_px = keypoints * image_size
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
```

**Step 3: Test imports**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.train import train; print('Training module imported')"`

Expected: `Training module imported`

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/dataset.py sprite_keypoint_detector/train.py
git commit -m "feat: add dataset and training pipeline for sprite keypoint detector"
```

---

### Task 5: Build Inference and Visualization

**Files:**
- Create: `sprite_keypoint_detector/inference.py`

**Step 1: Create inference.py**

Create `sprite_keypoint_detector/inference.py`:

```python
"""Inference utilities for sprite keypoint detector."""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2

from .model import SpriteKeypointDetector
from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS, SKELETON_CONNECTIONS, SKELETON_COLORS


class SpriteKeypointPredictor:
    """Wrapper for easy inference with trained model."""

    def __init__(self, model_path: Path, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        self.model = SpriteKeypointDetector(num_keypoints=NUM_KEYPOINTS)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {model_path}")
        if 'val_error_px' in checkpoint:
            print(f"  Validation error: {checkpoint['val_error_px']:.1f}px")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Union[Path, Image.Image, np.ndarray]) -> Dict[str, Tuple[int, int]]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        orig_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            keypoints_normalized = self.model(image_tensor)[0]

        keypoints_px = keypoints_normalized.cpu().numpy()
        keypoints_px[:, 0] *= orig_size[0]
        keypoints_px[:, 1] *= orig_size[1]

        result = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            result[name] = (int(round(keypoints_px[i, 0])), int(round(keypoints_px[i, 1])))

        return result


def draw_skeleton(
    image: Union[Image.Image, np.ndarray],
    keypoints: Dict[str, Tuple[int, int]],
    point_radius: int = 5,
    line_thickness: int = 2
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    result = image.copy()
    kp_list = [keypoints.get(name) for name in KEYPOINT_NAMES]

    for (i, j), color in zip(SKELETON_CONNECTIONS, SKELETON_COLORS):
        if kp_list[i] is not None and kp_list[j] is not None:
            cv2.line(result, kp_list[i], kp_list[j], color, line_thickness)

    for i, kp in enumerate(kp_list):
        if kp is not None:
            color = SKELETON_COLORS[min(i, len(SKELETON_COLORS)-1)]
            cv2.circle(result, kp, point_radius, color, -1)
            cv2.circle(result, kp, point_radius, (255, 255, 255), 1)

    return result


def render_skeleton_only(
    keypoints: Dict[str, Tuple[int, int]],
    image_size: Tuple[int, int] = (512, 512),
    point_radius: int = 5,
    line_thickness: int = 2,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    result = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
    return draw_skeleton(result, keypoints, point_radius, line_thickness)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m sprite_keypoint_detector.inference <model.pth> <image.png>")
        sys.exit(1)

    predictor = SpriteKeypointPredictor(Path(sys.argv[1]))
    keypoints = predictor.predict(Path(sys.argv[2]))

    print("Detected keypoints:")
    for name, (x, y) in keypoints.items():
        print(f"  {name}: ({x}, {y})")

    image = Image.open(sys.argv[2]).convert('RGB')
    result = draw_skeleton(image, keypoints)
    output_path = Path(sys.argv[2]).parent / f"{Path(sys.argv[2]).stem}_skeleton.png"
    Image.fromarray(result).save(output_path)
    print(f"\nSaved: {output_path}")
```

**Step 2: Test imports**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.inference import SpriteKeypointPredictor; print('Inference module imported')"`

Expected: `Inference module imported`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/inference.py
git commit -m "feat: add inference and visualization for sprite keypoint detector"
```

---

### Task 6: Create CLI Entry Point

**Files:**
- Create: `sprite_keypoint_detector/__main__.py`

**Step 1: Create __main__.py**

Create `sprite_keypoint_detector/__main__.py`:

```python
"""Command-line interface for sprite keypoint detector."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Sprite Keypoint Detector')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Annotate
    ann = subparsers.add_parser('annotate', help='Annotate keypoints')
    ann.add_argument('image_dir', type=Path)
    ann.add_argument('--output', '-o', type=Path, default=None)
    ann.add_argument('--seed', type=Path, default=None, help='Seed annotations JSON')
    ann.add_argument('--pattern', '-p', type=str, default='*.png')

    # Train
    tr = subparsers.add_parser('train', help='Train detector')
    tr.add_argument('annotations', type=Path)
    tr.add_argument('image_dir', type=Path)
    tr.add_argument('--output', '-o', type=Path, default=Path('models/sprite_keypoint'))
    tr.add_argument('--epochs', '-e', type=int, default=100)
    tr.add_argument('--batch-size', '-b', type=int, default=8)
    tr.add_argument('--lr', type=float, default=1e-3)
    tr.add_argument('--device', type=str, default=None)

    # Predict
    pr = subparsers.add_parser('predict', help='Detect keypoints')
    pr.add_argument('model', type=Path)
    pr.add_argument('images', type=Path, nargs='+')
    pr.add_argument('--output-dir', '-o', type=Path, default=None)
    pr.add_argument('--skeleton-only', action='store_true')

    args = parser.parse_args()

    if args.command == 'annotate':
        from .annotator import annotate_directory
        output = args.output or (args.image_dir / 'annotations.json')
        annotate_directory(args.image_dir, output, args.pattern, seed_json=args.seed)

    elif args.command == 'train':
        from .train import train
        train(args.annotations, args.image_dir, args.output,
              epochs=args.epochs, batch_size=args.batch_size,
              learning_rate=args.lr, device=args.device)

    elif args.command == 'predict':
        from .inference import SpriteKeypointPredictor, draw_skeleton, render_skeleton_only
        from PIL import Image

        predictor = SpriteKeypointPredictor(args.model)
        output_dir = args.output_dir or Path('.')
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in args.images:
            keypoints = predictor.predict(img_path)
            print(f"\n{img_path.name}:")
            for name, (x, y) in keypoints.items():
                print(f"  {name}: ({x}, {y})")

            if args.skeleton_only:
                result = render_skeleton_only(keypoints)
            else:
                image = Image.open(img_path).convert('RGB')
                result = draw_skeleton(image, keypoints)

            output_path = output_dir / f"{img_path.stem}_skeleton.png"
            Image.fromarray(result).save(output_path)
            print(f"  Saved: {output_path}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
```

**Step 2: Test CLI**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector --help`

Expected: Help output showing annotate, train, predict commands

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/__main__.py
git commit -m "feat: add CLI for sprite keypoint detector"
```

---

## Usage After Implementation

**1. Review seed annotations (AI-generated initial keypoints):**
```bash
cd /Users/roberthyatt/Code/ComfyUI
python3 -m sprite_keypoint_detector annotate training_data/frames -o training_data/annotations.json --seed training_data/seed_annotations.json
```

**2. Train the model:**
```bash
python3 -m sprite_keypoint_detector train training_data/annotations.json training_data/frames -o models/sprite_keypoint -e 100
```

**3. Predict on new images:**
```bash
python3 -m sprite_keypoint_detector predict models/sprite_keypoint/best_model.pth training_data/frames/*.png --skeleton-only
```
