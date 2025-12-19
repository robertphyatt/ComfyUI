# Palette-Based Color Synchronization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace body-segment color correction with SNES-style 16-color palette extraction and remapping to ensure all animation frames use identical colors.

**Architecture:** Extract optimal 16-color palette from all inpainted frames using k-means clustering, then remap every pixel to the nearest palette color. This eliminates pose-dependency issues and guarantees color consistency.

**Tech Stack:** Python, numpy, scikit-learn (KMeans), OpenCV

---

### Task 1: Replace color_correction.py with palette functions

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Replace entire file contents**

Replace the entire contents of `color_correction.py` with:

```python
"""Palette-based color synchronization for sprite animations.

Extracts an optimal N-color palette from all frames using k-means clustering,
then remaps every pixel to the nearest palette color. This ensures all frames
use identical colors regardless of pose differences.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.cluster import KMeans


def extract_palette(frames: List[np.ndarray], n_colors: int = 16) -> np.ndarray:
    """Extract optimal n-color palette from all frames using k-means.

    Args:
        frames: List of BGRA images
        n_colors: Number of colors in palette (default 16 for SNES-style)

    Returns:
        Palette array of shape (n_colors, 3) with BGR values
    """
    # Collect all visible pixels from all frames
    all_pixels = []
    for frame in frames:
        alpha = frame[:, :, 3]
        mask = alpha > 128
        bgr = frame[:, :, :3][mask]  # Shape: (N, 3)
        all_pixels.append(bgr)

    all_pixels = np.vstack(all_pixels)  # Shape: (total_pixels, 3)
    print(f"  Collected {len(all_pixels):,} pixels from {len(frames)} frames")

    # K-means clustering to find optimal palette
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(all_pixels)

    palette = kmeans.cluster_centers_.astype(np.uint8)  # Shape: (n_colors, 3)
    return palette


def remap_frame_to_palette(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Remap all visible pixels in frame to nearest palette color.

    Args:
        frame: BGRA image
        palette: Array of shape (n_colors, 3) with BGR values

    Returns:
        Remapped BGRA image
    """
    result = frame.copy()
    alpha = frame[:, :, 3]
    mask = alpha > 128

    # Get visible pixel coordinates
    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        pixel_bgr = frame[y, x, :3].astype(np.float32)

        # Find nearest palette color (Euclidean distance)
        distances = np.sqrt(np.sum((palette.astype(np.float32) - pixel_bgr) ** 2, axis=1))
        nearest_idx = np.argmin(distances)

        result[y, x, :3] = palette[nearest_idx]

    return result


def remap_all_frames(
    frames: List[np.ndarray],
    palette: np.ndarray
) -> List[np.ndarray]:
    """Remap all frames to use the shared palette.

    Args:
        frames: List of BGRA images
        palette: Array of shape (n_colors, 3) with BGR values

    Returns:
        List of remapped BGRA images
    """
    results = []
    for i, frame in enumerate(frames):
        remapped = remap_frame_to_palette(frame, palette)
        results.append(remapped)
        print(f"  Frame {i:02d}: remapped to palette")
    return results


def save_palette_image(palette: np.ndarray, path: Path) -> None:
    """Save palette as a visual swatch image.

    Creates a 4x4 grid of 32x32 color swatches.

    Args:
        palette: Array of shape (n_colors, 3) with BGR values
        path: Output path for the image
    """
    import cv2

    n_colors = len(palette)
    cols = 4
    rows = (n_colors + cols - 1) // cols
    swatch_size = 32

    img = np.zeros((rows * swatch_size, cols * swatch_size, 3), dtype=np.uint8)

    for i, color in enumerate(palette):
        row, col = i // cols, i % cols
        y1, y2 = row * swatch_size, (row + 1) * swatch_size
        x1, x2 = col * swatch_size, (col + 1) * swatch_size
        img[y1:y2, x1:x2] = color

    cv2.imwrite(str(path), img)
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import extract_palette, remap_all_frames; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "refactor(color_correction): replace body-segment with palette-based approach

Remove keypoint-relative positioning (caused black legs on pose differences).
New approach uses k-means to extract 16-color palette from all frames,
then remaps every pixel to nearest palette color.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Update pipeline.py to use palette approach

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Update imports**

Find (around line 46-47):
```python
from .golden_selection import select_golden_frame
from .color_correction import color_correct_all
```

Replace with:
```python
from .color_correction import extract_palette, remap_all_frames, save_palette_image
```

**Step 2: Update debug directory creation**

Find (around line 346):
```python
            (debug_dir / "5_color_corrected").mkdir(exist_ok=True)
```

Replace with:
```python
            (debug_dir / "5_palette_remapped").mkdir(exist_ok=True)
```

**Step 3: Replace golden selection and color correction block**

Find the block starting with (around line 418-445):
```python
        # === Golden frame selection and color correction ===
        print("\n=== Golden Frame Selection ===")
        print("Select the best frame as golden reference...")

        # Convert BGR to RGB for display in matplotlib GUI
        display_frames = []
        for frame in inpainted_frames:
            if frame.shape[2] == 4:
                # BGRA -> RGBA
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            else:
                # BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frames.append(rgb)

        # Show GUI for golden selection
        golden_idx = select_golden_frame(display_frames)
        print(f"Selected frame {frame_indices[golden_idx]:02d} as golden reference")

        # Color correct all frames
        print("\n=== Applying Color Correction ===")
        corrected_frames = color_correct_all(inpainted_frames, frame_keypoints, golden_idx)

        # Save color-corrected frames to debug if enabled
        if debug:
            for i, (corrected, base_idx) in enumerate(zip(corrected_frames, frame_indices)):
                cv2.imwrite(str(debug_dir / "5_color_corrected" / f"frame_{base_idx:02d}.png"), corrected)
            print(f"  Saved color-corrected frames to debug/5_color_corrected/")
```

Replace with:
```python
        # === Palette-based color synchronization ===
        print("\n=== Extracting Color Palette ===")
        palette = extract_palette(inpainted_frames, n_colors=16)
        print(f"  Extracted {len(palette)}-color palette")

        # Save palette visualization if debug enabled
        if debug:
            save_palette_image(palette, debug_dir / "palette.png")
            print(f"  Saved palette visualization to debug/palette.png")

        print("\n=== Remapping Frames to Palette ===")
        corrected_frames = remap_all_frames(inpainted_frames, palette)

        # Save palette-remapped frames to debug if enabled
        if debug:
            for i, (corrected, base_idx) in enumerate(zip(corrected_frames, frame_indices)):
                cv2.imwrite(str(debug_dir / "5_palette_remapped" / f"frame_{base_idx:02d}.png"), corrected)
            print(f"  Saved palette-remapped frames to debug/5_palette_remapped/")
```

**Step 4: Update comparison image step names**

Find (around line 65):
```python
        ("5_color_corrected", "Color Corrected"),
```

Replace with:
```python
        ("5_palette_remapped", "Palette"),
```

**Step 5: Remove unused frame_keypoints collection**

Find and remove these lines (around line 412-413):
```python
            # Store keypoints for color correction (use base keypoints since armor is aligned to base)
            frame_keypoints.append(base_kpts)
```

Also find and remove the initialization (around line 334):
```python
        frame_keypoints = []
```

**Step 6: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --help`

Expected: Help text displays without import errors

**Step 7: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): use palette-based color sync instead of golden frame

- Remove golden frame selection GUI
- Remove body-segment keypoint-relative correction
- Add 16-color palette extraction from all frames
- Remap all frames to shared palette
- Save palette.png visualization in debug output

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Delete golden_selection.py

**Files:**
- Delete: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/golden_selection.py`

**Step 1: Delete the file**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
rm golden_selection.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: remove golden_selection.py (no longer needed)

Palette-based approach doesn't require user to select a golden frame.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Test full pipeline

**Files:**
- None (testing only)

**Step 1: Run pipeline with debug output**

```bash
cd /Users/roberthyatt/Code/ComfyUI
rm -rf /tmp/pipeline_test
python3 -m sprite_keypoint_detector.pipeline \
  --frames-dir training_data/frames/ \
  --annotations training_data/annotations.json \
  --masks training_data/masks_corrected/ \
  --output /tmp/pipeline_test/ \
  --skip-validation \
  --debug
```

Expected:
1. Pipeline runs through transform steps
2. Extracts 16-color palette (no GUI popup)
3. Remaps all frames to palette
4. Pixelization runs
5. Output saved with debug directories

**Step 2: Verify debug output**

```bash
ls /tmp/pipeline_test/debug/
```

Expected: Should show `1_aligned/`, `2_masked/`, `3_rotated/`, `4_inpainted/`, `5_palette_remapped/`, `6_final/`, `comparison/`, `palette.png`

**Step 3: Check palette visualization**

```bash
open /tmp/pipeline_test/debug/palette.png
```

Expected: 4x4 grid showing the 16 extracted colors

**Step 4: Check comparison images**

```bash
open /tmp/pipeline_test/debug/comparison/frame_01.png
```

Expected: Side-by-side showing all steps including "Palette" step with consistent colors (no black legs!)

---

## Summary

This plan implements:
1. **Replace color_correction.py** - New palette-based functions (extract_palette, remap_all_frames)
2. **Update pipeline.py** - Remove golden selection, use palette approach
3. **Delete golden_selection.py** - No longer needed
4. **Test** - Verify end-to-end with debug output
