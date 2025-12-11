# Semi-Automated Mask Labeling Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a semi-automated tool to generate and manually correct pixel-level segmentation masks for training a custom clothing extraction model.

**Architecture:** Two-phase approach - (1) automated initial mask generation using color-based segmentation on difference images, (2) simple matplotlib-based GUI for manual correction with brush tools.

**Tech Stack:** Python, OpenCV, NumPy, Pillow, matplotlib for GUI, keyboard/mouse event handling

---

## Task 1: Automated Initial Mask Generation

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/generate_initial_masks.py`
- Create: `/Users/roberthyatt/Code/ComfyUI/tests/test_initial_masks.py`

**Step 1: Write the failing test**

Create test file:

```python
# tests/test_initial_masks.py
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
from generate_initial_masks import generate_mask_from_color_diff


def test_generate_mask_identifies_brown_armor_pixels():
    """Test that color-based mask generation identifies brown armor pixels."""
    # Create test images
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    base[200:300, 200:300] = [128, 128, 128]  # Gray head region

    clothed = base.copy()
    clothed[250:400, 150:350] = [101, 67, 33]  # Brown armor overlapping head

    mask = generate_mask_from_color_diff(base, clothed)

    # Verify mask shape
    assert mask.shape == (512, 512)
    assert mask.dtype == np.uint8

    # Verify armor region marked as 1
    assert mask[300, 250] == 1  # Below head, should be armor

    # Verify background marked as 0
    assert mask[100, 100] == 0  # Far from character
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/roberthyatt/Code/ComfyUI
pytest tests/test_initial_masks.py::test_generate_mask_identifies_brown_armor_pixels -v
```

Expected output: `ModuleNotFoundError: No module named 'generate_initial_masks'`

**Step 3: Write minimal implementation**

```python
# generate_initial_masks.py
"""Generate initial segmentation masks using color-based segmentation."""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def generate_mask_from_color_diff(base_img: np.ndarray, clothed_img: np.ndarray) -> np.ndarray:
    """Generate binary mask from color difference between base and clothed frames.

    Args:
        base_img: Base character image (H, W, 3) RGB uint8
        clothed_img: Clothed character image (H, W, 3) RGB uint8

    Returns:
        Binary mask (H, W) uint8, where 1=clothing, 0=not-clothing
    """
    # Compute absolute difference
    diff = cv2.absdiff(clothed_img, base_img)

    # Sum across RGB channels to get total change
    diff_magnitude = np.sum(diff, axis=2)

    # Threshold: pixels with significant change
    changed_pixels = diff_magnitude > 30

    # Analyze colors in changed regions
    mask = np.zeros(base_img.shape[:2], dtype=np.uint8)

    for y in range(base_img.shape[0]):
        for x in range(base_img.shape[1]):
            if not changed_pixels[y, x]:
                continue

            # Get pixel color in clothed image
            r, g, b = clothed_img[y, x]

            # Check if brown-ish (armor color range)
            # Brown: R > G > B, with R in range 80-140
            is_brown = (r > g and g > b and 80 <= r <= 140)

            # Check if gray-ish (base character head)
            # Gray: R ≈ G ≈ B
            color_variance = np.std([r, g, b])
            is_gray = color_variance < 20

            # Mark as clothing if brown and not gray
            if is_brown and not is_gray:
                mask[y, x] = 1

    return mask


def generate_all_masks(frames_dir: Path, output_dir: Path):
    """Generate initial masks for all 25 frame pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_num in range(25):
        base_path = frames_dir / f"base_frame_{frame_num:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_num:02d}.png"
        mask_path = output_dir / f"mask_{frame_num:02d}.png"

        if mask_path.exists():
            print(f"Frame {frame_num:02d}: Skipping (already exists)")
            continue

        # Load images
        base = np.array(Image.open(base_path).convert('RGB'))
        clothed = np.array(Image.open(clothed_path).convert('RGB'))

        # Generate mask
        mask = generate_mask_from_color_diff(base, clothed)

        # Save mask
        mask_img = Image.fromarray(mask * 255, 'L')
        mask_img.save(mask_path)

        # Statistics
        clothing_pixels = np.sum(mask == 1)
        percent = 100 * clothing_pixels / (512 * 512)
        print(f"Frame {frame_num:02d}: ✓ Generated ({clothing_pixels} pixels, {percent:.1f}%)")


if __name__ == "__main__":
    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/masks_initial")

    print("Generating initial masks using color-based segmentation...")
    print("=" * 70)
    generate_all_masks(frames_dir, output_dir)
    print("=" * 70)
    print(f"✓ Initial masks saved to {output_dir}/")
    print("\nNext: Use mask_correction_tool.py to fix any errors")
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/roberthyatt/Code/ComfyUI
pytest tests/test_initial_masks.py::test_generate_mask_identifies_brown_armor_pixels -v
```

Expected output: `PASSED`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add generate_initial_masks.py tests/test_initial_masks.py
git commit -m "feat: add color-based initial mask generation

- Implements difference-based detection with color thresholding
- Identifies brown armor vs gray head pixels
- Generates masks for all 25 training frames
- Test coverage for core mask generation logic"
```

---

## Task 2: Manual Mask Correction GUI

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/mask_correction_tool.py`
- Create: `/Users/roberthyatt/Code/ComfyUI/tests/test_mask_correction.py`

**Step 1: Write the failing test**

```python
# tests/test_mask_correction.py
import numpy as np
import pytest
from mask_correction_tool import MaskEditor


def test_mask_editor_initializes_with_images():
    """Test that MaskEditor loads images correctly."""
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    clothed = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)

    editor = MaskEditor(base, clothed, mask)

    assert editor.base_img.shape == (512, 512, 3)
    assert editor.clothed_img.shape == (512, 512, 3)
    assert editor.mask.shape == (512, 512)
    assert editor.brush_size > 0


def test_mask_editor_paint_adds_pixels():
    """Test that painting adds pixels to mask."""
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    clothed = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)

    editor = MaskEditor(base, clothed, mask)

    # Paint at position (100, 100) with brush size 5
    editor.paint_at(100, 100, value=1, brush_size=5)

    # Verify pixels were painted
    assert mask[100, 100] == 1
    assert mask[102, 102] == 1  # Within brush radius
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/roberthyatt/Code/ComfyUI
pytest tests/test_mask_correction.py -v
```

Expected output: `ModuleNotFoundError: No module named 'mask_correction_tool'`

**Step 3: Write minimal implementation**

```python
# mask_correction_tool.py
"""Interactive GUI tool for correcting segmentation masks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from pathlib import Path
from PIL import Image
import cv2


class MaskEditor:
    """Interactive mask editor with brush tool."""

    def __init__(self, base_img: np.ndarray, clothed_img: np.ndarray, mask: np.ndarray):
        """Initialize editor.

        Args:
            base_img: Base character image (512, 512, 3) RGB uint8
            clothed_img: Clothed character image (512, 512, 3) RGB uint8
            mask: Binary mask (512, 512) uint8 where 1=clothing
        """
        self.base_img = base_img
        self.clothed_img = clothed_img
        self.mask = mask.copy()
        self.original_mask = mask.copy()

        self.brush_size = 5
        self.paint_mode = 1  # 1=add clothing, 0=remove clothing
        self.is_painting = False

        # Setup figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('Mask Correction Tool - Left Click: Paint | Right Click: Erase | Scroll: Brush Size')

        # Display images
        self.axes[0].imshow(self.clothed_img)
        self.axes[0].set_title('Clothed Frame')
        self.axes[0].axis('off')

        self.axes[1].imshow(self.base_img)
        self.axes[1].set_title('Base Frame')
        self.axes[1].axis('off')

        self.mask_display = self.axes[2].imshow(self.get_overlay(), alpha=1.0)
        self.axes[2].set_title('Mask Overlay (Red=Clothing)')
        self.axes[2].axis('off')

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Add buttons
        self.add_buttons()

    def get_overlay(self) -> np.ndarray:
        """Create red overlay showing mask on clothed image."""
        overlay = self.clothed_img.copy().astype(np.float32) / 255.0

        # Red tint for masked regions
        overlay[self.mask == 1] = [1.0, 0.0, 0.0]

        return overlay

    def paint_at(self, x: int, y: int, value: int, brush_size: int):
        """Paint circular brush at position.

        Args:
            x: X coordinate
            y: Y coordinate
            value: 0 or 1 (remove or add clothing)
            brush_size: Brush radius in pixels
        """
        # Create circular brush
        for dy in range(-brush_size, brush_size + 1):
            for dx in range(-brush_size, brush_size + 1):
                if dx*dx + dy*dy <= brush_size*brush_size:
                    py, px = y + dy, x + dx
                    if 0 <= py < 512 and 0 <= px < 512:
                        self.mask[py, px] = value

    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if event.inaxes != self.axes[2]:
            return

        self.is_painting = True

        # Left click = add clothing (1), Right click = remove (0)
        self.paint_mode = 1 if event.button == 1 else 0

        x, y = int(event.xdata), int(event.ydata)
        self.paint_at(x, y, self.paint_mode, self.brush_size)
        self.update_display()

    def on_mouse_up(self, event):
        """Handle mouse button release."""
        self.is_painting = False

    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if not self.is_painting or event.inaxes != self.axes[2]:
            return

        x, y = int(event.xdata), int(event.ydata)
        self.paint_at(x, y, self.paint_mode, self.brush_size)
        self.update_display()

    def on_scroll(self, event):
        """Handle scroll to change brush size."""
        if event.button == 'up':
            self.brush_size = min(50, self.brush_size + 2)
        else:
            self.brush_size = max(1, self.brush_size - 2)

        self.fig.suptitle(f'Brush Size: {self.brush_size} | Left Click: Paint | Right Click: Erase')
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'r':  # Reset to original
            self.mask = self.original_mask.copy()
            self.update_display()
        elif event.key == 'c':  # Clear all
            self.mask.fill(0)
            self.update_display()

    def update_display(self):
        """Refresh mask overlay display."""
        self.mask_display.set_data(self.get_overlay())
        self.fig.canvas.draw_idle()

    def add_buttons(self):
        """Add save/cancel buttons."""
        # Save button
        save_ax = plt.axes([0.7, 0.01, 0.1, 0.04])
        save_btn = Button(save_ax, 'Save')
        save_btn.on_clicked(lambda e: self.save_and_close())

        # Cancel button
        cancel_ax = plt.axes([0.82, 0.01, 0.1, 0.04])
        cancel_btn = Button(cancel_ax, 'Cancel')
        cancel_btn.on_clicked(lambda e: plt.close())

    def save_and_close(self):
        """Mark as saved and close."""
        self.saved = True
        plt.close()

    def show(self) -> np.ndarray:
        """Show editor and return corrected mask."""
        self.saved = False
        plt.show()
        return self.mask if self.saved else None


def correct_mask_interactive(frame_num: int, frames_dir: Path, masks_dir: Path, output_dir: Path):
    """Interactively correct a single mask.

    Args:
        frame_num: Frame number (0-24)
        frames_dir: Directory containing frame images
        masks_dir: Directory containing initial masks
        output_dir: Directory to save corrected masks
    """
    # Load images
    base_path = frames_dir / f"base_frame_{frame_num:02d}.png"
    clothed_path = frames_dir / f"clothed_frame_{frame_num:02d}.png"
    mask_path = masks_dir / f"mask_{frame_num:02d}.png"

    base = np.array(Image.open(base_path).convert('RGB'))
    clothed = np.array(Image.open(clothed_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L')) // 255

    # Edit mask
    editor = MaskEditor(base, clothed, mask)
    corrected_mask = editor.show()

    if corrected_mask is not None:
        # Save corrected mask
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"mask_{frame_num:02d}.png"

        mask_img = Image.fromarray(corrected_mask * 255, 'L')
        mask_img.save(output_path)

        # Statistics
        clothing_pixels = np.sum(corrected_mask == 1)
        percent = 100 * clothing_pixels / (512 * 512)
        print(f"Frame {frame_num:02d}: ✓ Saved ({clothing_pixels} pixels, {percent:.1f}%)")
        return True
    else:
        print(f"Frame {frame_num:02d}: Cancelled")
        return False


def correct_all_masks(frames_dir: Path, initial_masks_dir: Path, corrected_masks_dir: Path):
    """Correct all 25 masks interactively."""
    print("Mask Correction Tool")
    print("=" * 70)
    print("Controls:")
    print("  - Left Click: Add clothing pixels (paint red)")
    print("  - Right Click: Remove clothing pixels (erase red)")
    print("  - Scroll: Adjust brush size")
    print("  - 'R' key: Reset to initial mask")
    print("  - 'C' key: Clear all (start from scratch)")
    print("=" * 70)
    print()

    for frame_num in range(25):
        output_path = corrected_masks_dir / f"mask_{frame_num:02d}.png"

        if output_path.exists():
            response = input(f"Frame {frame_num:02d} already corrected. Re-edit? (y/N): ")
            if response.lower() != 'y':
                print(f"Frame {frame_num:02d}: Skipping")
                continue

        print(f"\nEditing Frame {frame_num:02d}/24...")
        success = correct_mask_interactive(frame_num, frames_dir, initial_masks_dir, corrected_masks_dir)

        if not success:
            print("Correction cancelled. Exiting.")
            break

    print()
    print("=" * 70)
    print(f"✓ Corrected masks saved to {corrected_masks_dir}/")


if __name__ == "__main__":
    frames_dir = Path("training_data/frames")
    initial_masks_dir = Path("training_data/masks_initial")
    corrected_masks_dir = Path("training_data/masks_corrected")

    correct_all_masks(frames_dir, initial_masks_dir, corrected_masks_dir)
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/roberthyatt/Code/ComfyUI
pytest tests/test_mask_correction.py -v
```

Expected output: `PASSED` (both tests)

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add mask_correction_tool.py tests/test_mask_correction.py
git commit -m "feat: add interactive mask correction GUI

- Matplotlib-based editor with brush tool
- Left/right click to add/remove clothing pixels
- Scroll to adjust brush size
- Keyboard shortcuts for reset/clear
- Processes all 25 frames sequentially
- Test coverage for core painting logic"
```

---

## Task 3: Integration Test and Usage Documentation

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/tests/test_mask_workflow.py`
- Create: `/Users/roberthyatt/Code/ComfyUI/docs/mask_labeling_workflow.md`

**Step 1: Write integration test**

```python
# tests/test_mask_workflow.py
"""Integration test for complete mask generation and correction workflow."""

import shutil
from pathlib import Path
import pytest
from generate_initial_masks import generate_all_masks


def test_complete_workflow_generates_all_masks(tmp_path):
    """Test that workflow generates 25 initial masks."""
    # Setup test directories
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()

    # Copy test frames
    source_frames = Path("training_data/frames")
    for frame_num in range(25):
        shutil.copy(
            source_frames / f"base_frame_{frame_num:02d}.png",
            frames_dir / f"base_frame_{frame_num:02d}.png"
        )
        shutil.copy(
            source_frames / f"clothed_frame_{frame_num:02d}.png",
            frames_dir / f"clothed_frame_{frame_num:02d}.png"
        )

    # Generate masks
    generate_all_masks(frames_dir, masks_dir)

    # Verify all 25 masks created
    mask_files = list(masks_dir.glob("mask_*.png"))
    assert len(mask_files) == 25

    # Verify mask properties
    from PIL import Image
    mask = Image.open(masks_dir / "mask_00.png")
    assert mask.size == (512, 512)
    assert mask.mode == 'L'
```

**Step 2: Run test**

```bash
cd /Users/roberthyatt/Code/ComfyUI
pytest tests/test_mask_workflow.py -v
```

Expected output: `PASSED`

**Step 3: Create usage documentation**

```markdown
# docs/mask_labeling_workflow.md

# Mask Labeling Workflow

Complete workflow for generating training data for the custom segmentation model.

## Prerequisites

- 25 base frames in `training_data/frames/base_frame_XX.png`
- 25 clothed frames in `training_data/frames/clothed_frame_XX.png`

## Step 1: Generate Initial Masks (Automated)

Run automated color-based mask generation:

\`\`\`bash
cd /Users/roberthyatt/Code/ComfyUI
python generate_initial_masks.py
\`\`\`

Output: `training_data/masks_initial/mask_XX.png` (25 files)

This uses color thresholding to identify brown armor pixels vs gray head pixels.

## Step 2: Correct Masks (Manual)

Launch the interactive correction tool:

\`\`\`bash
python mask_correction_tool.py
\`\`\`

**Controls:**
- **Left Click:** Add clothing pixels (paint red overlay)
- **Right Click:** Remove clothing pixels (erase red)
- **Mouse Scroll:** Adjust brush size (1-50 pixels)
- **'R' Key:** Reset to initial automated mask
- **'C' Key:** Clear all (start from scratch)
- **Save Button:** Save corrected mask and move to next frame
- **Cancel Button:** Discard changes and exit

**Workflow per frame:**
1. Review the red overlay on the clothed frame
2. Use left click to paint areas that should be clothing
3. Use right click to erase areas that shouldn't be clothing
4. Focus on:
   - Gray head pixels (should NOT be red)
   - Brown armor pixels (SHOULD be red)
   - Edge accuracy around armor boundaries
5. Click Save when satisfied
6. Tool automatically loads next frame

Estimated time: 5-10 minutes per frame = 2-3 hours total

## Step 3: Verify Results

Check corrected masks:

\`\`\`bash
ls -l training_data/masks_corrected/
# Should show 25 files: mask_00.png through mask_24.png
\`\`\`

Spot check quality:

\`\`\`bash
# Open a few masks to verify
open training_data/masks_corrected/mask_00.png
open training_data/masks_corrected/mask_12.png
open training_data/masks_corrected/mask_24.png
\`\`\`

Look for:
- Clean edges around armor
- No red on gray head
- All armor areas covered

## Output

Final training data structure:

\`\`\`
training_data/
├── frames/
│   ├── base_frame_00.png ... base_frame_24.png
│   └── clothed_frame_00.png ... clothed_frame_24.png
├── masks_initial/          # Automated generation (reference)
│   └── mask_00.png ... mask_24.png
└── masks_corrected/        # Manual corrections (TRAINING DATA)
    └── mask_00.png ... mask_24.png
\`\`\`

Use `masks_corrected/` for training the U-Net model.
```

**Step 4: Commit documentation**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add tests/test_mask_workflow.py docs/mask_labeling_workflow.md
git commit -m "docs: add mask labeling workflow documentation

- Integration test for complete workflow
- Step-by-step usage guide
- Tool controls reference
- Expected output structure"
```

---

## Verification Steps

After completing all tasks:

1. **Run all tests:**
   ```bash
   cd /Users/roberthyatt/Code/ComfyUI
   pytest tests/test_initial_masks.py tests/test_mask_correction.py tests/test_mask_workflow.py -v
   ```
   Expected: All tests PASS

2. **Generate initial masks:**
   ```bash
   python generate_initial_masks.py
   ```
   Expected: 25 mask files in `training_data/masks_initial/`

3. **Test correction tool on one frame:**
   ```bash
   # Edit just frame 0 to verify GUI works
   python -c "
from mask_correction_tool import correct_mask_interactive
from pathlib import Path
correct_mask_interactive(0, Path('training_data/frames'), Path('training_data/masks_initial'), Path('training_data/masks_corrected'))
"
   ```
   Expected: GUI opens, allows editing, saves to `training_data/masks_corrected/mask_00.png`

4. **Check git status:**
   ```bash
   git status
   ```
   Expected: Clean working tree (all changes committed)

---

## Next Steps

After completing this plan:

1. **Label all 25 frames** using the correction tool (~2-3 hours)
2. **Implement U-Net model** (separate plan needed)
3. **Train on labeled data** (10-30 minutes)
4. **Integrate into extraction pipeline** (replace Ollama step)

---

## Notes

- **Color ranges may need tuning:** If initial masks are poor quality, adjust brown/gray thresholds in `generate_mask_from_color_diff()` function
- **Brush size defaults to 5 pixels:** Good for detail work; increase for bulk filling
- **Save frequently:** Tool saves one frame at a time; corrections are persistent
- **Re-edit anytime:** Can reopen any frame to make corrections
