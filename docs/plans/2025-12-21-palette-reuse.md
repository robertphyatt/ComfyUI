# Palette Reuse Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow reusing a palette from a previous pipeline run to ensure color consistency across multiple animations for the same armor.

**Architecture:** New `--palette-from` CLI argument loads palette from previous output directory. Standalone utility extracts palette from any spritesheet for recovery.

**Tech Stack:** Python, NumPy, OpenCV, existing color_correction.py infrastructure

---

## Task 1: Add load_palette_from_image function

**Files:**
- Modify: `sprite_keypoint_detector/color_correction.py`

**Step 1: Add load_palette_from_image function**

```python
def load_palette_from_image(path: Path) -> np.ndarray:
    """Load palette from a swatch image.

    Parses a 4x4 grid of 32x32 color swatches back to palette array.

    Args:
        path: Path to palette swatch image (128x128 PNG)

    Returns:
        Palette array of shape (16, 3) with BGR values

    Raises:
        FileNotFoundError: If palette image doesn't exist
        ValueError: If image format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Palette not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read palette image: {path}")

    if img.shape[0] < 128 or img.shape[1] < 128:
        raise ValueError(f"Palette image too small: {img.shape}, expected at least 128x128")

    # Sample center of each 32x32 swatch cell
    swatch_size = 32
    palette = []
    for row in range(4):
        for col in range(4):
            center_y = row * swatch_size + swatch_size // 2
            center_x = col * swatch_size + swatch_size // 2
            color = img[center_y, center_x]
            palette.append(color)

    return np.array(palette, dtype=np.uint8)
```

**Step 2: Run test**

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.color_correction import load_palette_from_image, save_palette_image
from pathlib import Path
import numpy as np

# Create test palette
test_palette = np.array([
    [i * 16, i * 16, i * 16] for i in range(16)
], dtype=np.uint8)

# Save it
save_palette_image(test_palette, Path('/tmp/test_palette.png'))

# Load it back
loaded = load_palette_from_image(Path('/tmp/test_palette.png'))
assert loaded.shape == (16, 3), f'Wrong shape: {loaded.shape}'
assert np.allclose(loaded, test_palette, atol=1), 'Colors dont match'
print('load_palette_from_image OK')
"
```

Expected: `load_palette_from_image OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/color_correction.py
git commit -m "feat: add load_palette_from_image function"
```

---

## Task 2: Add --palette-from argument to pipeline

**Files:**
- Modify: `sprite_keypoint_detector/pipeline.py`

**Step 1: Add argument to argparse**

Find the argument parser section and add after `--pixelize`:

```python
parser.add_argument("--palette-from", type=Path,
                   help="Reuse palette from previous output directory (loads debug/palette.png)")
```

**Step 2: Pass palette_from to pipeline**

Find where `ClothingPipeline` is instantiated in `main()` and update it to pass `palette_from` to the pipeline. Add `palette_from` parameter to `__init__` and store it.

In `ClothingPipeline.__init__`, add parameter:
```python
palette_from: Optional[Path] = None
```

And store it:
```python
self.palette_from = palette_from
```

**Step 3: Update generate_outputs to use palette_from**

In `generate_outputs()`, find the palette extraction section. Replace it with:

```python
# === Extract or Load Palette ===
if self.palette_from:
    print(f"\n=== Loading Palette from {self.palette_from} ===")
    palette_path = self.palette_from / "debug" / "palette.png"
    from .color_correction import load_palette_from_image
    global_palette = load_palette_from_image(palette_path)
    print(f"  Loaded {len(global_palette)}-color palette from {palette_path}")
else:
    print("\n=== Extracting Global Palette ===")
    # ... existing extraction code ...
```

**Step 4: Run import test**

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.pipeline import ClothingPipeline
print('Pipeline import OK')
"
```

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/pipeline.py
git commit -m "feat(pipeline): add --palette-from to reuse palette from previous run"
```

---

## Task 3: Create extract_palette utility

**Files:**
- Create: `sprite_keypoint_detector/extract_palette.py`

**Step 1: Create the utility module**

```python
"""Extract palette from a spritesheet image.

Usage:
    python -m sprite_keypoint_detector.extract_palette <spritesheet.png> <output_palette.png>
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from .color_correction import extract_palette, save_palette_image


def extract_palette_from_spritesheet(
    spritesheet_path: Path,
    n_colors: int = 16
) -> np.ndarray:
    """Extract palette from a spritesheet image.

    Args:
        spritesheet_path: Path to spritesheet image
        n_colors: Number of colors to extract (default 16)

    Returns:
        Palette array of shape (n_colors, 3) with BGR values
    """
    img = cv2.imread(str(spritesheet_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {spritesheet_path}")

    # Use existing extract_palette function (expects list of frames)
    # Treat entire spritesheet as single frame
    if len(img.shape) == 2:
        # Grayscale - convert to BGRA
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        # BGR - add alpha channel (fully opaque)
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)

    return extract_palette([img], n_colors)


def main():
    parser = argparse.ArgumentParser(
        description="Extract palette from a spritesheet image"
    )
    parser.add_argument("spritesheet", type=Path,
                       help="Path to spritesheet image")
    parser.add_argument("output", type=Path,
                       help="Path to save palette image")
    parser.add_argument("--colors", type=int, default=16,
                       help="Number of colors to extract (default 16)")

    args = parser.parse_args()

    if not args.spritesheet.exists():
        print(f"Error: Spritesheet not found: {args.spritesheet}")
        return 1

    print(f"Extracting {args.colors}-color palette from {args.spritesheet}...")
    palette = extract_palette_from_spritesheet(args.spritesheet, args.colors)

    print(f"Saving palette to {args.output}...")
    save_palette_image(palette, args.output)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
```

**Step 2: Run test**

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.extract_palette \
    training_data/clothing_spritesheet.png \
    /tmp/extracted_palette.png && \
    ls -la /tmp/extracted_palette.png
```

Expected: Creates palette file

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/extract_palette.py
git commit -m "feat: add extract_palette utility for palette recovery"
```

---

## Task 4: Integration test

**Step 1: Run pipeline with --palette-from**

```bash
cd /Users/roberthyatt/Code/ComfyUI

# First run creates palette
python3 -u -m sprite_keypoint_detector.pipeline \
    --frames-dir training_data/frames \
    --annotations training_data/annotations.json \
    --masks training_data/masks_corrected \
    --output /tmp/first_run \
    --debug \
    --skip-validation

# Second run reuses palette
python3 -u -m sprite_keypoint_detector.pipeline \
    --frames-dir training_data/frames \
    --annotations training_data/annotations.json \
    --masks training_data/masks_corrected \
    --output /tmp/second_run \
    --palette-from /tmp/first_run \
    --debug \
    --skip-validation
```

**Step 2: Verify palettes match**

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
import cv2
import numpy as np
p1 = cv2.imread('/tmp/first_run/debug/palette.png')
p2 = cv2.imread('/tmp/second_run/debug/palette.png')
if np.array_equal(p1, p2):
    print('Palettes match!')
else:
    print('ERROR: Palettes differ')
"
```

Expected: `Palettes match!`

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add load_palette_from_image function |
| 2 | Add --palette-from argument to pipeline |
| 3 | Create extract_palette utility |
| 4 | Integration test |
