# Frame-to-Frame Pixel Consistency Checker

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate frame-to-frame pixel consistency in animation sequences by tracking where each pixel "should" move based on joint positions.

**Architecture:** Use joint-based tracking with translation + rotation transforms per bone segment. Compare expected vs actual pixel colors using palette index distance. Generate consistency mask visualization added to comparison strips.

**Tech Stack:** Python, NumPy, OpenCV, existing keypoint annotations

---

## Overview & Data Flow

```
Frame N + Frame N+1 + Keypoints
         |
    For each bone segment:
      1. Get pixels in segment region (frame N)
      2. Compute transform (translation + rotation) from joints
      3. Warp expected positions to frame N+1
         |
    For each warped pixel position:
      1. Compare expected color vs actual color
      2. Compute palette index distance
         |
    Generate consistency mask:
      - White: correct color
      - Red gradient: wrong color (intensity = palette distance)
      - Gray: untracked (not in any segment)
         |
    Add as new column in comparison strip
```

**25 comparisons:** 0→1, 1→2, ... 23→24, 24→0 (loop closure)

---

## Bone Segments

From keypoint annotations:

| Segment | Joint A (idx) | Joint B (idx) |
|---------|---------------|---------------|
| Head/Neck | neck (1) | head (0) |
| L upper arm | l_shoulder (2) | l_elbow (4) |
| L forearm | l_elbow (4) | l_wrist (6) |
| R upper arm | r_shoulder (3) | r_elbow (5) |
| R forearm | r_elbow (5) | r_wrist (7) |
| L thigh | l_hip (8) | l_knee (10) |
| L shin | l_knee (10) | l_ankle (12) |
| R thigh | r_hip (9) | r_knee (11) |
| R shin | r_knee (11) | r_ankle (13) |

---

## Transform Computation

For each segment from joint A to joint B:

```python
# Frame N:   A_n, B_n (joint positions)
# Frame N+1: A_n1, B_n1

# Translation: movement of segment midpoint
mid_n = (A_n + B_n) / 2
mid_n1 = (A_n1 + B_n1) / 2
translation = mid_n1 - mid_n

# Rotation: angle change of segment
angle_n = atan2(B_n.y - A_n.y, B_n.x - A_n.x)
angle_n1 = atan2(B_n1.y - A_n1.y, B_n1.x - A_n1.x)
rotation = angle_n1 - angle_n

# Apply: rotate pixel around segment midpoint, then translate
```

---

## Consistency Mask Visualization

```python
# For each pixel in frame N that belongs to a tracked segment:
expected_pos = warp_pixel(pixel_pos, segment_transform)

if expected_pos out of bounds:
    continue  # Skip - don't mark in mask

if expected_pos in frame N+1 transparent area:
    color = BRIGHT_RED  # Pixel should exist but doesn't (max error)
else:
    actual_color = frame_n1[expected_pos]
    expected_color = frame_n[pixel_pos]

    if actual_color == expected_color:
        color = WHITE  # Correct
    else:
        # Palette index distance (0-15 range)
        expected_idx = find_palette_index(expected_color)
        actual_idx = find_palette_index(actual_color)
        distance = abs(expected_idx - actual_idx)

        # Map distance to red intensity (1-15 -> light to dark red)
        red_intensity = 255 - (distance * 16)
        color = (0, 0, red_intensity)  # BGR dark red gradient

# Untracked pixels (not in any segment): GRAY (128, 128, 128)
```

---

## Implementation

### Task 1: Create consistency.py with segment transform functions

**Files:**
- Create: `sprite_keypoint_detector/consistency.py`

**Step 1: Write segment transform computation**

```python
"""Frame-to-frame pixel consistency checker."""

import numpy as np
from typing import Tuple, List
import cv2
from math import atan2, cos, sin


# Bone segments: (joint_a_idx, joint_b_idx, name)
BONE_SEGMENTS = [
    (1, 0, "head_neck"),
    (2, 4, "l_upper_arm"),
    (4, 6, "l_forearm"),
    (3, 5, "r_upper_arm"),
    (5, 7, "r_forearm"),
    (8, 10, "l_thigh"),
    (10, 12, "l_shin"),
    (9, 11, "r_thigh"),
    (11, 13, "r_shin"),
]


def compute_segment_transform(
    joint_a_n: np.ndarray,
    joint_b_n: np.ndarray,
    joint_a_n1: np.ndarray,
    joint_b_n1: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute translation and rotation between frame N and N+1 for a segment.

    Args:
        joint_a_n: Joint A position in frame N (x, y)
        joint_b_n: Joint B position in frame N (x, y)
        joint_a_n1: Joint A position in frame N+1 (x, y)
        joint_b_n1: Joint B position in frame N+1 (x, y)

    Returns:
        (translation, rotation_angle, center_n) where:
        - translation: (dx, dy) movement of segment midpoint
        - rotation_angle: angle change in radians
        - center_n: midpoint in frame N (rotation center)
    """
    # Midpoints
    mid_n = (joint_a_n + joint_b_n) / 2
    mid_n1 = (joint_a_n1 + joint_b_n1) / 2

    # Translation
    translation = mid_n1 - mid_n

    # Rotation angles
    angle_n = atan2(joint_b_n[1] - joint_a_n[1], joint_b_n[0] - joint_a_n[0])
    angle_n1 = atan2(joint_b_n1[1] - joint_a_n1[1], joint_b_n1[0] - joint_a_n1[0])
    rotation = angle_n1 - angle_n

    return translation, rotation, mid_n


def warp_pixel_position(
    pixel_pos: Tuple[int, int],
    center: np.ndarray,
    translation: np.ndarray,
    rotation: float
) -> Tuple[int, int]:
    """Warp a pixel position from frame N to expected position in frame N+1.

    Args:
        pixel_pos: (x, y) position in frame N
        center: rotation center (segment midpoint in frame N)
        translation: (dx, dy) translation
        rotation: rotation angle in radians

    Returns:
        (x, y) expected position in frame N+1
    """
    # Translate to origin (center)
    px, py = pixel_pos[0] - center[0], pixel_pos[1] - center[1]

    # Rotate
    cos_r, sin_r = cos(rotation), sin(rotation)
    rx = px * cos_r - py * sin_r
    ry = px * sin_r + py * cos_r

    # Translate back and apply movement
    new_x = rx + center[0] + translation[0]
    new_y = ry + center[1] + translation[1]

    return int(round(new_x)), int(round(new_y))
```

**Step 2: Run basic test**

```bash
cd /Users/roberthyatt/Code/ComfyUI
python3 -c "from sprite_keypoint_detector.consistency import compute_segment_transform, warp_pixel_position; print('Import OK')"
```

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/consistency.py
git commit -m "feat: add segment transform functions for frame consistency"
```

---

### Task 2: Add segment mask and consistency mask generation

**Files:**
- Modify: `sprite_keypoint_detector/consistency.py`

**Step 1: Add segment pixel collection and consistency mask generation**

```python
def get_segment_pixels(
    frame: np.ndarray,
    keypoints: np.ndarray,
    joint_a_idx: int,
    joint_b_idx: int,
    segment_width: int = 50
) -> np.ndarray:
    """Get boolean mask of pixels belonging to a segment.

    Args:
        frame: RGBA frame
        keypoints: Keypoint array
        joint_a_idx: Index of joint A
        joint_b_idx: Index of joint B
        segment_width: Width of segment region

    Returns:
        Boolean mask of pixels in segment
    """
    h, w = frame.shape[:2]
    joint_a = keypoints[joint_a_idx]
    joint_b = keypoints[joint_b_idx]

    # Create capsule-shaped region around segment
    mask = np.zeros((h, w), dtype=bool)

    # Vector along segment
    seg_vec = joint_b - joint_a
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1:
        return mask
    seg_unit = seg_vec / seg_len

    # Perpendicular vector
    perp = np.array([-seg_unit[1], seg_unit[0]])

    # For each pixel, check if within segment region
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    points = np.stack([x_coords, y_coords], axis=-1).astype(float)

    # Vector from joint_a to each point
    to_point = points - joint_a

    # Project onto segment
    along = np.sum(to_point * seg_unit, axis=-1)
    across = np.abs(np.sum(to_point * perp, axis=-1))

    # Within segment bounds
    in_segment = (along >= 0) & (along <= seg_len) & (across <= segment_width / 2)

    # Intersect with visible pixels
    visible = frame[:, :, 3] > 128

    return in_segment & visible


def find_palette_index(color: np.ndarray, palette: np.ndarray) -> int:
    """Find the index of a color in the palette.

    Args:
        color: BGR color (3,)
        palette: Palette array (n_colors, 3)

    Returns:
        Index of nearest palette color
    """
    distances = np.sqrt(np.sum((palette.astype(float) - color.astype(float)) ** 2, axis=1))
    return int(np.argmin(distances))


def generate_consistency_mask(
    frame_n: np.ndarray,
    frame_n1: np.ndarray,
    keypoints_n: np.ndarray,
    keypoints_n1: np.ndarray,
    palette: np.ndarray,
    segment_width: int = 50
) -> np.ndarray:
    """Generate consistency mask comparing frame N to frame N+1.

    Args:
        frame_n: Current frame RGBA
        frame_n1: Next frame RGBA
        keypoints_n: Keypoints for frame N
        keypoints_n1: Keypoints for frame N+1
        palette: Color palette (n_colors, 3) BGR
        segment_width: Width for segment regions

    Returns:
        Consistency mask (BGR image):
        - White (255,255,255): correct color
        - Red gradient: wrong color (darker = more different)
        - Gray (128,128,128): untracked pixel
        - Black (0,0,0): transparent in frame N
    """
    h, w = frame_n.shape[:2]
    mask = np.full((h, w, 3), 128, dtype=np.uint8)  # Default gray (untracked)

    # Track which pixels have been processed
    tracked = np.zeros((h, w), dtype=bool)

    # Process each bone segment
    for joint_a_idx, joint_b_idx, name in BONE_SEGMENTS:
        # Get segment pixels in frame N
        segment_mask = get_segment_pixels(
            frame_n, keypoints_n, joint_a_idx, joint_b_idx, segment_width
        )

        # Skip if no pixels
        if not np.any(segment_mask):
            continue

        # Compute transform
        translation, rotation, center = compute_segment_transform(
            keypoints_n[joint_a_idx],
            keypoints_n[joint_b_idx],
            keypoints_n1[joint_a_idx],
            keypoints_n1[joint_b_idx]
        )

        # Check each pixel in segment
        ys, xs = np.where(segment_mask)
        for y, x in zip(ys, xs):
            if tracked[y, x]:
                continue  # Already processed by another segment
            tracked[y, x] = True

            # Warp to expected position
            expected_x, expected_y = warp_pixel_position(
                (x, y), center, translation, rotation
            )

            # Check bounds
            if expected_x < 0 or expected_x >= w or expected_y < 0 or expected_y >= h:
                continue  # Out of bounds - skip

            # Get colors
            expected_color = frame_n[y, x, :3]

            # Check if position is transparent in frame N+1
            if frame_n1[expected_y, expected_x, 3] < 128:
                mask[y, x] = [0, 0, 255]  # Bright red - pixel missing
                continue

            actual_color = frame_n1[expected_y, expected_x, :3]

            # Compare colors
            if np.array_equal(expected_color, actual_color):
                mask[y, x] = [255, 255, 255]  # White - correct
            else:
                # Palette index distance
                expected_idx = find_palette_index(expected_color, palette)
                actual_idx = find_palette_index(actual_color, palette)
                distance = abs(expected_idx - actual_idx)

                # Map to red intensity (distance 1-15 -> 239 down to 15)
                red_intensity = max(15, 255 - (distance * 16))
                mask[y, x] = [0, 0, red_intensity]  # BGR red gradient

    # Mark transparent pixels in frame N as black
    transparent_n = frame_n[:, :, 3] < 128
    mask[transparent_n] = [0, 0, 0]

    return mask
```

**Step 2: Test with sample frames**

```bash
python3 -c "
from sprite_keypoint_detector.consistency import generate_consistency_mask
import numpy as np
print('Consistency mask generation OK')
"
```

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/consistency.py
git commit -m "feat: add consistency mask generation with palette distance"
```

---

### Task 3: Integrate into pipeline comparison output

**Files:**
- Modify: `sprite_keypoint_detector/pipeline.py`

**Step 1: Import consistency module**

Add to imports at top of pipeline.py:
```python
from .consistency import generate_consistency_mask
```

**Step 2: Add consistency mask to comparison strip generation**

Find the comparison strip generation code and add consistency mask column. The exact location will need to be identified - look for where `debug/comparison/` images are saved.

**Step 3: Test full pipeline with consistency output**

```bash
python3 -u -m sprite_keypoint_detector.pipeline \
    --frames-dir training_data/frames \
    --annotations training_data/annotations.json \
    --masks training_data/masks_corrected \
    --output /tmp/pipeline_consistency_test \
    --debug \
    --skip-validation
```

**Step 4: Verify consistency column appears in comparison images**

```bash
open /tmp/pipeline_consistency_test/debug/comparison/frame_00.png
```

**Step 5: Commit**

```bash
git add sprite_keypoint_detector/pipeline.py
git commit -m "feat: add frame consistency visualization to comparison output"
```

---

### Task 4: Add loop closure comparison (frame 24 → frame 0)

**Files:**
- Modify: `sprite_keypoint_detector/pipeline.py`

**Step 1: Generate consistency mask for frame 24 → 0**

After generating all sequential comparisons, add one more for loop closure.

**Step 2: Save as separate image or append to frame 24's comparison**

**Step 3: Commit**

```bash
git add sprite_keypoint_detector/pipeline.py
git commit -m "feat: add loop closure consistency check (frame 24 to 0)"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Create consistency.py with segment transform functions |
| 2 | Add segment mask and consistency mask generation |
| 3 | Integrate into pipeline comparison output |
| 4 | Add loop closure comparison |
