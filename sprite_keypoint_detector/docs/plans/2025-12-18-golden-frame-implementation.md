# Golden Frame Color Correction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add golden frame selection GUI and body-segment-based color correction to ensure consistent pixel colors across all animation frames.

**Architecture:** After existing pipeline (align→mask→rotate→inpaint), show GUI for user to select best frame, then map all other frames' pixel colors to match the golden frame using keypoint-relative positions within body segments. Pixelization happens last.

**Tech Stack:** Python, numpy, matplotlib (GUI), scipy.ndimage (spatial operations)

---

### Task 1: Remove pixelization from transform_frame functions

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`

**Step 1: Update transform_frame to skip pixelization**

In `transform_frame()` (around line 599-602), change:

```python
    # Step 4: Pixelize
    final_armor = apply_pixelize(inpainted_armor, config.pixelize_factor)

    return final_armor
```

To:

```python
    # Pixelization now happens after color correction in pipeline
    return inpainted_armor
```

**Step 2: Update transform_frame_debug to skip pixelization**

In `transform_frame_debug()`, find where `final_armor` is assigned (around line 654) and change:

```python
    # Step 4: Pixelize
    final_armor = apply_pixelize(inpainted_armor, config.pixelize_factor)
```

To:

```python
    # Pixelization now happens after color correction in pipeline
    final_armor = inpainted_armor
```

**Step 3: Update TransformDebugOutput docstring**

Change the docstring for `final_armor` field (around line 457):

```python
    final_armor: np.ndarray           # After inpaint (pre-pixelize)
```

**Step 4: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import transform_frame; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "refactor(transform): remove pixelization from transform functions

Pixelization will now happen after color correction in the pipeline.
apply_pixelize() remains available as standalone function.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Create body segment definitions

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Create color_correction.py with segment definitions**

```python
"""Color correction using golden frame reference.

Maps pixel colors from a golden reference frame to all other frames
using body-segment-based keypoint-relative positioning.
"""

import numpy as np
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class BodySegment(IntEnum):
    """Body segment identifiers."""
    HEAD = 0
    TORSO = 1
    LEFT_UPPER_ARM = 2
    LEFT_LOWER_ARM = 3
    RIGHT_UPPER_ARM = 4
    RIGHT_LOWER_ARM = 5
    LEFT_UPPER_LEG = 6
    LEFT_LOWER_LEG = 7
    RIGHT_UPPER_LEG = 8
    RIGHT_LOWER_LEG = 9


# Keypoint indices for each segment (start, end)
# Based on keypoints.py: 0=head, 1=neck, 2=left_shoulder, 3=right_shoulder,
# 4=left_elbow, 5=right_elbow, 6=left_wrist, 7=right_wrist,
# 8=left_fingertip, 9=right_fingertip, 10=left_hip, 11=right_hip,
# 12=left_knee, 13=right_knee, 14=left_ankle, 15=right_ankle,
# 16=left_toe, 17=right_toe
SEGMENT_KEYPOINTS: Dict[BodySegment, Tuple[int, int]] = {
    BodySegment.HEAD: (0, 1),           # head -> neck
    BodySegment.TORSO: (1, 10),         # neck -> left_hip (use as torso anchor)
    BodySegment.LEFT_UPPER_ARM: (2, 4),  # left_shoulder -> left_elbow
    BodySegment.LEFT_LOWER_ARM: (4, 6),  # left_elbow -> left_wrist
    BodySegment.RIGHT_UPPER_ARM: (3, 5), # right_shoulder -> right_elbow
    BodySegment.RIGHT_LOWER_ARM: (5, 7), # right_elbow -> right_wrist
    BodySegment.LEFT_UPPER_LEG: (10, 12), # left_hip -> left_knee
    BodySegment.LEFT_LOWER_LEG: (12, 14), # left_knee -> left_ankle
    BodySegment.RIGHT_UPPER_LEG: (11, 13), # right_hip -> right_knee
    BodySegment.RIGHT_LOWER_LEG: (13, 15), # right_knee -> right_ankle
}


@dataclass
class PixelPosition:
    """Relative position of a pixel within a body segment."""
    segment: BodySegment
    along_bone: float      # 0.0 = at start keypoint, 1.0 = at end keypoint
    perpendicular: float   # signed distance perpendicular to bone (pixels)
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import BodySegment, SEGMENT_KEYPOINTS; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add body segment definitions

Define 10 body segments with their keypoint pairs for
segment-based pixel mapping.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Implement pixel-to-segment assignment

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Add segment assignment function**

Add after the `PixelPosition` dataclass:

```python
def _point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> Tuple[float, float]:
    """Compute distance from point to line segment.

    Returns:
        (along_bone, perpendicular): Position along bone (0-1) and signed perpendicular distance
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        # Degenerate line segment
        return 0.5, np.linalg.norm(point - line_start)

    line_unit = line_vec / line_len
    point_vec = point - line_start

    # Project point onto line
    along = np.dot(point_vec, line_unit) / line_len  # 0-1 range (can be outside)

    # Perpendicular distance (signed: positive = left of bone direction)
    perp_vec = point_vec - (along * line_len) * line_unit
    perp_dist = np.linalg.norm(perp_vec)

    # Sign: use cross product to determine which side
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    if cross < 0:
        perp_dist = -perp_dist

    return along, perp_dist


def assign_pixel_to_segment(
    pixel_y: int,
    pixel_x: int,
    keypoints: np.ndarray
) -> Optional[PixelPosition]:
    """Assign a pixel to its nearest body segment.

    Args:
        pixel_y: Pixel Y coordinate
        pixel_x: Pixel X coordinate
        keypoints: 18x2 array of keypoint coordinates

    Returns:
        PixelPosition with segment and relative position, or None if too far from any segment
    """
    point = np.array([pixel_x, pixel_y], dtype=np.float64)

    best_segment = None
    best_distance = float('inf')
    best_along = 0.0
    best_perp = 0.0

    for segment, (kp_start, kp_end) in SEGMENT_KEYPOINTS.items():
        start = keypoints[kp_start]
        end = keypoints[kp_end]

        along, perp = _point_to_line_distance(point, start, end)

        # Distance to segment (clamped along to 0-1 for distance calc)
        clamped_along = max(0.0, min(1.0, along))
        closest_on_segment = start + clamped_along * (end - start)
        distance = np.linalg.norm(point - closest_on_segment)

        if distance < best_distance:
            best_distance = distance
            best_segment = segment
            best_along = along
            best_perp = perp

    if best_segment is None:
        return None

    return PixelPosition(
        segment=best_segment,
        along_bone=best_along,
        perpendicular=best_perp
    )
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import assign_pixel_to_segment; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add pixel-to-segment assignment

Assigns each pixel to nearest body segment and computes
relative position (along bone, perpendicular distance).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Implement golden frame lookup structure

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Add golden frame index builder**

Add after `assign_pixel_to_segment`:

```python
@dataclass
class GoldenPixel:
    """A pixel from the golden frame with its position and color."""
    y: int
    x: int
    rgb: np.ndarray  # (3,) uint8
    position: PixelPosition


def build_golden_index(
    golden_frame: np.ndarray,
    golden_keypoints: np.ndarray
) -> Dict[BodySegment, List[GoldenPixel]]:
    """Build lookup structure for golden frame pixels by segment.

    Args:
        golden_frame: RGBA image of golden frame
        golden_keypoints: 18x2 keypoints for golden frame

    Returns:
        Dict mapping each segment to list of its pixels with positions
    """
    h, w = golden_frame.shape[:2]
    alpha = golden_frame[:, :, 3]

    index: Dict[BodySegment, List[GoldenPixel]] = {seg: [] for seg in BodySegment}

    # Find all visible pixels and assign to segments
    visible_ys, visible_xs = np.where(alpha > 128)

    for y, x in zip(visible_ys, visible_xs):
        position = assign_pixel_to_segment(y, x, golden_keypoints)
        if position is not None:
            pixel = GoldenPixel(
                y=y,
                x=x,
                rgb=golden_frame[y, x, :3].copy(),
                position=position
            )
            index[position.segment].append(pixel)

    return index
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import build_golden_index; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add golden frame index builder

Indexes all visible pixels in golden frame by body segment
with their relative positions for fast lookup.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Implement color lookup from golden frame

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Add color lookup function**

Add after `build_golden_index`:

```python
def find_golden_color(
    position: PixelPosition,
    golden_index: Dict[BodySegment, List[GoldenPixel]],
    golden_keypoints: np.ndarray
) -> Optional[np.ndarray]:
    """Find the color from golden frame for a given relative position.

    Args:
        position: Relative position within a segment
        golden_index: Pre-built index of golden frame
        golden_keypoints: Keypoints for golden frame

    Returns:
        RGB color (3,) uint8, or None if not found
    """
    segment_pixels = golden_index.get(position.segment, [])

    if not segment_pixels:
        # Segment not visible in golden frame - try nearest segment
        return _find_nearest_segment_color(position, golden_index, golden_keypoints)

    # Find pixel with closest relative position
    best_pixel = None
    best_distance = float('inf')

    for pixel in segment_pixels:
        # Distance in relative position space
        along_diff = position.along_bone - pixel.position.along_bone
        perp_diff = position.perpendicular - pixel.position.perpendicular

        # Weight along_bone more since it's normalized 0-1
        distance = (along_diff * 50) ** 2 + perp_diff ** 2

        if distance < best_distance:
            best_distance = distance
            best_pixel = pixel

    if best_pixel is None:
        return None

    return best_pixel.rgb


def _find_nearest_segment_color(
    position: PixelPosition,
    golden_index: Dict[BodySegment, List[GoldenPixel]],
    golden_keypoints: np.ndarray
) -> Optional[np.ndarray]:
    """Fallback: find color from nearest segment that has pixels."""
    # Try segments in order of likely proximity
    segment_order = [
        BodySegment.TORSO,  # Most likely to have pixels
        BodySegment.LEFT_UPPER_ARM, BodySegment.RIGHT_UPPER_ARM,
        BodySegment.LEFT_UPPER_LEG, BodySegment.RIGHT_UPPER_LEG,
        BodySegment.LEFT_LOWER_ARM, BodySegment.RIGHT_LOWER_ARM,
        BodySegment.LEFT_LOWER_LEG, BodySegment.RIGHT_LOWER_LEG,
        BodySegment.HEAD,
    ]

    for seg in segment_order:
        if seg == position.segment:
            continue
        pixels = golden_index.get(seg, [])
        if pixels:
            # Return color from middle of this segment
            mid_idx = len(pixels) // 2
            return pixels[mid_idx].rgb

    return None
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import find_golden_color; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add golden color lookup

Finds matching color from golden frame by relative position
within body segment. Falls back to nearest segment if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Implement single frame color correction

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Add frame color correction function**

Add after `_find_nearest_segment_color`:

```python
def color_correct_frame(
    frame: np.ndarray,
    frame_keypoints: np.ndarray,
    golden_index: Dict[BodySegment, List[GoldenPixel]],
    golden_keypoints: np.ndarray
) -> np.ndarray:
    """Color correct a single frame using golden frame colors.

    Args:
        frame: RGBA image to correct
        frame_keypoints: 18x2 keypoints for this frame
        golden_index: Pre-built index of golden frame
        golden_keypoints: Keypoints for golden frame

    Returns:
        Color-corrected RGBA image
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    alpha = frame[:, :, 3]

    # Process all visible pixels
    visible_ys, visible_xs = np.where(alpha > 128)

    for y, x in zip(visible_ys, visible_xs):
        # Find this pixel's relative position
        position = assign_pixel_to_segment(y, x, frame_keypoints)

        if position is None:
            continue

        # Find matching color from golden frame
        golden_rgb = find_golden_color(position, golden_index, golden_keypoints)

        if golden_rgb is not None:
            result[y, x, :3] = golden_rgb

    return result
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import color_correct_frame; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add single frame color correction

Corrects all visible pixels in a frame to match golden frame
colors based on body-segment relative positions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Implement batch color correction

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Add batch color correction function**

Add after `color_correct_frame`:

```python
def color_correct_all(
    frames: List[np.ndarray],
    keypoints_list: List[np.ndarray],
    golden_idx: int
) -> List[np.ndarray]:
    """Color correct all frames using the golden frame as reference.

    Args:
        frames: List of RGBA images
        keypoints_list: List of 18x2 keypoint arrays (one per frame)
        golden_idx: Index of the golden frame

    Returns:
        List of color-corrected RGBA images
    """
    if not frames:
        return []

    # Build golden frame index
    golden_frame = frames[golden_idx]
    golden_keypoints = keypoints_list[golden_idx]
    golden_index = build_golden_index(golden_frame, golden_keypoints)

    # Color correct each frame
    results = []
    for i, (frame, kpts) in enumerate(zip(frames, keypoints_list)):
        if i == golden_idx:
            # Golden frame stays unchanged
            results.append(frame.copy())
        else:
            corrected = color_correct_frame(frame, kpts, golden_index, golden_keypoints)
            results.append(corrected)

    return results
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import color_correct_all; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add batch color correction

Orchestrates color correction across all frames using
golden frame as reference.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Create golden frame selection GUI

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/golden_selection.py`

**Step 1: Create golden_selection.py with GUI**

```python
"""GUI for selecting the golden reference frame."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import List


def select_golden_frame(frames: List[np.ndarray]) -> int:
    """Show GUI for user to select the golden reference frame.

    Args:
        frames: List of RGBA images to choose from

    Returns:
        Index of selected golden frame
    """
    if not frames:
        raise ValueError("No frames provided")

    if len(frames) == 1:
        return 0

    n_frames = len(frames)
    selected_idx = [0]  # Use list to allow modification in nested function
    confirmed = [False]

    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Select Golden Frame - Click thumbnail, then 'Select as Golden'", fontsize=14)

    # Left side: 5x5 thumbnail grid
    # Right side: Large preview
    # Bottom: Button

    # Calculate grid dimensions
    grid_cols = 5
    grid_rows = (n_frames + grid_cols - 1) // grid_cols

    # Create axes for thumbnails
    thumb_axes = []
    for i in range(n_frames):
        row = i // grid_cols
        col = i % grid_cols
        # Thumbnails take left 60% of figure, arranged in grid
        ax = fig.add_axes([
            0.02 + col * 0.11,  # x position
            0.85 - row * 0.16,  # y position (top to bottom)
            0.10,  # width
            0.14   # height
        ])
        ax.imshow(frames[i])
        ax.set_title(f"{i:02d}", fontsize=8)
        ax.axis('off')
        thumb_axes.append(ax)

    # Large preview on right
    preview_ax = fig.add_axes([0.60, 0.15, 0.38, 0.75])
    preview_img = preview_ax.imshow(frames[0])
    preview_ax.set_title(f"Frame 00", fontsize=12)
    preview_ax.axis('off')

    # Highlight current selection in thumbnails
    highlights = []
    for ax in thumb_axes:
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             fill=False, edgecolor='yellow', linewidth=3, visible=False)
        ax.add_patch(rect)
        highlights.append(rect)
    highlights[0].set_visible(True)

    # Button
    button_ax = fig.add_axes([0.70, 0.02, 0.20, 0.06])
    button = Button(button_ax, 'Select as Golden')

    def on_thumbnail_click(event):
        if event.inaxes in thumb_axes:
            idx = thumb_axes.index(event.inaxes)
            selected_idx[0] = idx

            # Update preview
            preview_img.set_data(frames[idx])
            preview_ax.set_title(f"Frame {idx:02d}", fontsize=12)

            # Update highlights
            for i, h in enumerate(highlights):
                h.set_visible(i == idx)

            fig.canvas.draw_idle()

    def on_button_click(event):
        confirmed[0] = True
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_thumbnail_click)
    button.on_clicked(on_button_click)

    # Also support keyboard navigation
    def on_key(event):
        if event.key == 'enter':
            confirmed[0] = True
            plt.close(fig)
        elif event.key == 'left':
            new_idx = max(0, selected_idx[0] - 1)
            selected_idx[0] = new_idx
            preview_img.set_data(frames[new_idx])
            preview_ax.set_title(f"Frame {new_idx:02d}", fontsize=12)
            for i, h in enumerate(highlights):
                h.set_visible(i == new_idx)
            fig.canvas.draw_idle()
        elif event.key == 'right':
            new_idx = min(n_frames - 1, selected_idx[0] + 1)
            selected_idx[0] = new_idx
            preview_img.set_data(frames[new_idx])
            preview_ax.set_title(f"Frame {new_idx:02d}", fontsize=12)
            for i, h in enumerate(highlights):
                h.set_visible(i == new_idx)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return selected_idx[0]
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.golden_selection import select_golden_frame; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add golden_selection.py
git commit -m "feat(golden_selection): add GUI for selecting golden frame

Matplotlib-based GUI with thumbnail grid, large preview,
and keyboard/mouse navigation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Integrate into pipeline

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Add imports at top of pipeline.py**

After the existing imports (around line 45), add:

```python
from .golden_selection import select_golden_frame
from .color_correction import color_correct_all
from .transform import apply_pixelize
```

**Step 2: Find the main processing loop and modify**

In `run_pipeline()` or the main processing function, after all frames are transformed but before saving, add the golden selection and color correction steps.

Find where the frames are collected after transform (look for where `transform_frame` or `transform_frame_debug` results are stored). After that loop completes, add:

```python
    # === NEW: Golden frame selection and color correction ===
    print("\n=== Golden Frame Selection ===")
    print("Select the best frame as golden reference...")

    # Collect pre-pixelized frames and keypoints for color correction
    pre_pixelize_frames = [...]  # The inpainted frames before pixelization
    frame_keypoints = [...]  # The keypoints for each frame

    # Show GUI for golden selection
    golden_idx = select_golden_frame(pre_pixelize_frames)
    print(f"Selected frame {golden_idx:02d} as golden reference")

    # Color correct all frames
    print("Applying color correction...")
    corrected_frames = color_correct_all(pre_pixelize_frames, frame_keypoints, golden_idx)

    # Apply pixelization as final step
    print("Applying pixelization...")
    final_frames = [apply_pixelize(f, config.pixelize_factor) for f in corrected_frames]
```

**Note:** The exact integration depends on how the pipeline currently stores intermediate results. The implementer should trace through the pipeline to find where frames and keypoints are accumulated.

**Step 3: Verify pipeline runs**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --help`

Expected: Help text displays without import errors

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): integrate golden selection and color correction

Pipeline now:
1. Runs existing transform steps (no pixelization)
2. Shows GUI for golden frame selection
3. Color corrects all frames to match golden
4. Applies pixelization as final step

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Update debug output directories

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Add new debug output directories**

Find where debug directories are created (look for `1_aligned`, `2_masked`, etc.) and add:

```python
    # Create new debug directories
    if debug_dir:
        (debug_dir / "5_color_corrected").mkdir(exist_ok=True)
        (debug_dir / "6_final").mkdir(exist_ok=True)
```

**Step 2: Save color-corrected and final frames to debug**

After color correction and pixelization, save to debug directories:

```python
    # Save debug outputs
    if debug_dir:
        for i, (corrected, final) in enumerate(zip(corrected_frames, final_frames)):
            # Save color-corrected (pre-pixelize)
            corrected_path = debug_dir / "5_color_corrected" / f"frame_{i:02d}.png"
            cv2.imwrite(str(corrected_path), cv2.cvtColor(corrected, cv2.COLOR_RGBA2BGRA))

            # Save final (post-pixelize)
            final_path = debug_dir / "6_final" / f"frame_{i:02d}.png"
            cv2.imwrite(str(final_path), cv2.cvtColor(final, cv2.COLOR_RGBA2BGRA))
```

**Step 3: Update comparison image creation**

Find `create_debug_comparison` function and update the `steps` list to include new steps:

```python
    steps = [
        ("1_aligned", "Aligned"),
        ("2_masked", "Masked"),
        ("3_rotated", "Rotated"),
        ("4_inpainted", "Inpainted"),
        ("5_color_corrected", "Color Corrected"),
        ("6_final", "Final"),
    ]
```

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): add debug output for color correction steps

New debug directories:
- 5_color_corrected/: After golden frame color correction
- 6_final/: After pixelization

Comparison images now include all steps.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Test full pipeline

**Files:**
- None (testing only)

**Step 1: Run pipeline with debug output**

```bash
cd /Users/roberthyatt/Code/ComfyUI
python3 -m sprite_keypoint_detector.pipeline \
  --frames-dir training_data/frames/ \
  --annotations training_data/annotations.json \
  --masks training_data/masks_corrected/ \
  --output /tmp/pipeline_test/ \
  --debug
```

Expected:
1. Pipeline runs through transform steps
2. GUI pops up showing all frames
3. User selects golden frame, clicks button
4. Color correction runs
5. Pixelization runs
6. Output saved with debug directories including `5_color_corrected/` and `6_final/`

**Step 2: Verify debug output**

```bash
ls /tmp/pipeline_test/debug/
```

Expected: Should show `1_aligned/`, `2_masked/`, `3_rotated/`, `4_inpainted/`, `5_color_corrected/`, `6_final/`, `comparison/`

**Step 3: Check comparison images**

```bash
open /tmp/pipeline_test/debug/comparison/frame_00.png
```

Expected: Side-by-side comparison showing all pipeline steps including color correction

**Step 4: Push all changes**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git push
```

---

## Summary

This plan implements:
1. **Remove pixelization from transform** - Moved to after color correction
2. **Body segment definitions** - 10 segments mapped to keypoint pairs
3. **Pixel-to-segment assignment** - Maps each pixel to nearest segment with relative position
4. **Golden frame indexing** - Builds lookup structure for fast color matching
5. **Color lookup** - Finds matching colors from golden frame with nearest-neighbor fallback
6. **Frame color correction** - Corrects single frame's colors
7. **Batch color correction** - Orchestrates all frames
8. **Golden selection GUI** - Matplotlib interactive selector
9. **Pipeline integration** - Adds new steps after existing transform
10. **Debug output** - New directories for intermediate results
11. **Testing** - End-to-end verification
