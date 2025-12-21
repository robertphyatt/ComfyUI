# Color Consensus Correction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix frame-to-frame color inconsistencies by majority voting on palette indices across all 25 frames, then correcting outliers.

**Architecture:** Map pixels to canonical segment-relative coordinates, vote on consensus palette index per position across all frames, apply corrections. Uses existing BONE_SEGMENTS from consistency.py.

**Tech Stack:** Python, NumPy, collections.Counter, existing consistency.py infrastructure

---

## Overview

```
final_frames[] + keypoints[] + palette
        |
   Phase 1: Map pixels to canonical coords per segment
   Phase 2: Vote on palette index per canonical position (plurality wins)
   Phase 3: Apply corrections where pixel != consensus
        |
   corrected_frames[]
```

---

## Task 1: Create consensus.py with canonical coordinate functions

**Files:**
- Create: `sprite_keypoint_detector/consensus.py`

**Step 1: Create module with imports and docstring**

```python
"""Color consensus correction using majority voting across frames.

Maps pixels to canonical segment-relative coordinates, votes on the correct
palette index for each position, and corrects outliers to match consensus.
"""

import numpy as np
from typing import Dict, List, Tuple
from .consistency import BONE_SEGMENTS, find_palette_index
```

**Step 2: Add pixel_to_canonical function**

```python
def pixel_to_canonical(
    pixel_pos: Tuple[int, int],
    joint_a: np.ndarray,
    joint_b: np.ndarray
) -> Tuple[float, float]:
    """Convert pixel position to canonical segment coordinates.

    Canonical coords: x=0 at joint_a, x=1 at joint_b, y=perpendicular distance
    normalized by segment length.

    Args:
        pixel_pos: (x, y) pixel position
        joint_a: Segment start joint position
        joint_b: Segment end joint position

    Returns:
        (canonical_x, canonical_y) normalized coordinates
    """
    seg_vec = joint_b - joint_a
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1:
        return (0.0, 0.0)

    seg_unit = seg_vec / seg_len
    perp = np.array([-seg_unit[1], seg_unit[0]])

    to_pixel = np.array(pixel_pos, dtype=float) - joint_a
    along = np.dot(to_pixel, seg_unit) / seg_len
    across = np.dot(to_pixel, perp) / seg_len

    return (float(along), float(across))
```

**Step 3: Add discretize_canonical function**

```python
def discretize_canonical(
    x: float,
    y: float,
    resolution: float = 0.05
) -> Tuple[int, int]:
    """Round canonical coords to discrete grid position.

    Args:
        x: Canonical x coordinate
        y: Canonical y coordinate
        resolution: Grid resolution (default 0.05 = 20 bins per unit)

    Returns:
        (grid_x, grid_y) discrete grid position
    """
    return (int(round(x / resolution)), int(round(y / resolution)))
```

**Step 4: Run import test**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.consensus import pixel_to_canonical, discretize_canonical
import numpy as np

# Test canonical coords - horizontal segment
joint_a = np.array([10.0, 10.0])
joint_b = np.array([30.0, 10.0])  # Length 20, horizontal
pixel = (20, 10)  # Midpoint
cx, cy = pixel_to_canonical(pixel, joint_a, joint_b)
assert abs(cx - 0.5) < 0.01, f'Expected cx=0.5, got {cx}'
assert abs(cy - 0.0) < 0.01, f'Expected cy=0.0, got {cy}'

# Test discretization
gx, gy = discretize_canonical(0.52, 0.03)
assert gx == 10, f'Expected gx=10, got {gx}'
assert gy == 1, f'Expected gy=1, got {gy}'

print('canonical coordinate functions OK')
"
```

Expected: `canonical coordinate functions OK`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/consensus.py
git commit -m "feat: add canonical coordinate functions for consensus correction"
```

---

## Task 2: Add segment pixel collection with grid positions

**Files:**
- Modify: `sprite_keypoint_detector/consensus.py`

**Step 1: Add get_segment_pixels_with_positions function**

```python
def get_segment_pixels_with_positions(
    frame: np.ndarray,
    keypoints: np.ndarray,
    joint_a_idx: int,
    joint_b_idx: int,
    segment_width: int = 50
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Get visible pixels in segment with their canonical grid positions.

    Args:
        frame: RGBA frame
        keypoints: Keypoint array (14 joints)
        joint_a_idx: Index of joint A
        joint_b_idx: Index of joint B
        segment_width: Width of segment region in pixels

    Returns:
        List of ((pixel_x, pixel_y), (grid_x, grid_y)) tuples
    """
    h, w = frame.shape[:2]
    joint_a = keypoints[joint_a_idx]
    joint_b = keypoints[joint_b_idx]

    seg_vec = joint_b - joint_a
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1:
        return []

    seg_unit = seg_vec / seg_len
    perp = np.array([-seg_unit[1], seg_unit[0]])

    results = []
    half_width = segment_width / 2

    for y in range(h):
        for x in range(w):
            if frame[y, x, 3] <= 128:
                continue

            to_pixel = np.array([x, y], dtype=float) - joint_a
            along = np.dot(to_pixel, seg_unit)
            across = np.dot(to_pixel, perp)

            if along < 0 or along > seg_len or abs(across) > half_width:
                continue

            cx = along / seg_len
            cy = across / seg_len
            gx, gy = discretize_canonical(cx, cy)

            results.append(((x, y), (gx, gy)))

    return results
```

**Step 2: Run test**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.consensus import get_segment_pixels_with_positions
import numpy as np

# Create test frame with some visible pixels
frame = np.zeros((64, 64, 4), dtype=np.uint8)
frame[30:35, 20:40, :] = [255, 128, 64, 255]  # Horizontal stripe

# Keypoints for horizontal segment
keypoints = np.zeros((14, 2))
keypoints[2] = [20, 32]  # l_shoulder
keypoints[4] = [40, 32]  # l_elbow

results = get_segment_pixels_with_positions(frame, keypoints, 2, 4, segment_width=20)
print(f'Found {len(results)} pixels in segment')
assert len(results) > 0, 'Should find pixels in segment'
print('get_segment_pixels_with_positions OK')
"
```

Expected: `Found N pixels in segment` and `get_segment_pixels_with_positions OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/consensus.py
git commit -m "feat: add segment pixel collection with grid positions"
```

---

## Task 3: Add consensus map building

**Files:**
- Modify: `sprite_keypoint_detector/consensus.py`

**Step 1: Add build_consensus_map function**

```python
from collections import Counter


def build_consensus_map(
    frames: List[np.ndarray],
    keypoints_per_frame: List[np.ndarray],
    palette: np.ndarray,
    segment_width: int = 50
) -> Dict[Tuple[int, int, int], int]:
    """Build consensus palette index for each segment grid position.

    Collects votes from all frames, returns plurality winner for each position.

    Args:
        frames: List of RGBA frames
        keypoints_per_frame: Keypoints for each frame
        palette: Color palette (n_colors, 3) BGR
        segment_width: Width for segment regions

    Returns:
        Dict mapping (segment_idx, grid_x, grid_y) -> consensus palette index
    """
    votes: Dict[Tuple[int, int, int], List[int]] = {}

    for frame, keypoints in zip(frames, keypoints_per_frame):
        for seg_idx, (joint_a_idx, joint_b_idx, name) in enumerate(BONE_SEGMENTS):
            pixels_with_positions = get_segment_pixels_with_positions(
                frame, keypoints, joint_a_idx, joint_b_idx, segment_width
            )

            for (px, py), (gx, gy) in pixels_with_positions:
                key = (seg_idx, gx, gy)
                color = frame[py, px, :3]
                palette_idx = find_palette_index(color, palette)

                if key not in votes:
                    votes[key] = []
                votes[key].append(palette_idx)

    consensus_map = {}
    for key, idx_list in votes.items():
        counter = Counter(idx_list)
        winner = counter.most_common(1)[0][0]
        consensus_map[key] = winner

    return consensus_map
```

**Step 2: Run test**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.consensus import build_consensus_map
import numpy as np

# Create 3 test frames with same geometry
frames = []
keypoints_list = []
for i in range(3):
    frame = np.zeros((64, 64, 4), dtype=np.uint8)
    frame[30:35, 20:40, :] = [100, 100, 100, 255]
    frames.append(frame)

    keypoints = np.zeros((14, 2))
    keypoints[2] = [20, 32]
    keypoints[4] = [40, 32]
    keypoints_list.append(keypoints)

# Simple 4-color palette
palette = np.array([[0,0,0], [100,100,100], [200,200,200], [255,255,255]], dtype=np.uint8)

consensus = build_consensus_map(frames, keypoints_list, palette)
print(f'Consensus map has {len(consensus)} positions')
assert len(consensus) > 0, 'Should have consensus positions'
print('build_consensus_map OK')
"
```

Expected: `Consensus map has N positions` and `build_consensus_map OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/consensus.py
git commit -m "feat: add consensus map building with plurality voting"
```

---

## Task 4: Add consensus correction application

**Files:**
- Modify: `sprite_keypoint_detector/consensus.py`

**Step 1: Add apply_consensus_correction function**

```python
def apply_consensus_correction(
    frames: List[np.ndarray],
    keypoints_per_frame: List[np.ndarray],
    palette: np.ndarray,
    consensus_map: Dict[Tuple[int, int, int], int],
    segment_width: int = 50
) -> Tuple[List[np.ndarray], int]:
    """Apply consensus colors to all frames.

    Args:
        frames: List of RGBA frames
        keypoints_per_frame: Keypoints for each frame
        palette: Color palette (n_colors, 3) BGR
        consensus_map: Mapping from (segment_idx, grid_x, grid_y) to palette index
        segment_width: Width for segment regions

    Returns:
        (corrected_frames, total_corrections) tuple
    """
    corrected = [frame.copy() for frame in frames]
    total_corrections = 0

    for frame_idx, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
        processed = set()

        for seg_idx, (joint_a_idx, joint_b_idx, name) in enumerate(BONE_SEGMENTS):
            pixels_with_positions = get_segment_pixels_with_positions(
                frame, keypoints, joint_a_idx, joint_b_idx, segment_width
            )

            for (px, py), (gx, gy) in pixels_with_positions:
                if (px, py) in processed:
                    continue
                processed.add((px, py))

                key = (seg_idx, gx, gy)
                if key not in consensus_map:
                    continue

                consensus_idx = consensus_map[key]
                current_color = frame[py, px, :3]
                current_idx = find_palette_index(current_color, palette)

                if current_idx != consensus_idx:
                    corrected[frame_idx][py, px, :3] = palette[consensus_idx]
                    total_corrections += 1

    return corrected, total_corrections
```

**Step 2: Run test**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.consensus import build_consensus_map, apply_consensus_correction
import numpy as np

# Create 3 frames: 2 with color index 1, 1 with color index 2
palette = np.array([[0,0,0], [100,100,100], [200,200,200], [255,255,255]], dtype=np.uint8)

frames = []
keypoints_list = []
for i in range(3):
    frame = np.zeros((64, 64, 4), dtype=np.uint8)
    color = palette[1] if i < 2 else palette[2]  # Frame 2 is different
    frame[30:35, 20:40, :3] = color
    frame[30:35, 20:40, 3] = 255
    frames.append(frame)

    keypoints = np.zeros((14, 2))
    keypoints[2] = [20, 32]
    keypoints[4] = [40, 32]
    keypoints_list.append(keypoints)

consensus = build_consensus_map(frames, keypoints_list, palette)
corrected, num_corrections = apply_consensus_correction(frames, keypoints_list, palette, consensus)

print(f'Made {num_corrections} corrections')
assert num_corrections > 0, 'Should have made corrections to frame 2'

# Verify frame 2 now matches consensus
frame2_color = corrected[2][32, 30, :3]
assert np.array_equal(frame2_color, palette[1]), f'Frame 2 should be corrected to palette[1]'
print('apply_consensus_correction OK')
"
```

Expected: `Made N corrections` and `apply_consensus_correction OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/consensus.py
git commit -m "feat: add consensus correction application"
```

---

## Task 5: Integrate into pipeline

**Files:**
- Modify: `sprite_keypoint_detector/pipeline.py`

**Step 1: Add import**

Add after the consistency import (around line 47):

```python
from .consensus import build_consensus_map, apply_consensus_correction
```

**Step 2: Add to comparison strip steps**

In `create_debug_comparison()`, add to the `steps` list after `("7_consistency", "Consistency")`:

```python
("8_consensus", "Consensus"),
```

**Step 3: Add consensus correction in generate_outputs**

In `generate_outputs()`, after the consistency mask generation block (after the print statement about generated consistency masks), add:

```python
# === Apply Color Consensus Correction ===
print("\n=== Applying Color Consensus Correction ===")

# Collect keypoints for all frames (reuse from consistency generation)
all_keypoints = []
for base_idx in frame_indices:
    base_name = f"base_frame_{base_idx:02d}.png"
    keypoints = get_keypoints_array(base_annotations[base_name].get("keypoints", {}))
    all_keypoints.append(keypoints)

# Build consensus and apply corrections
consensus_map = build_consensus_map(final_frames, all_keypoints, global_palette)
print(f"  Built consensus map with {len(consensus_map)} positions")

final_frames, num_corrections = apply_consensus_correction(
    final_frames, all_keypoints, global_palette, consensus_map
)
print(f"  Applied {num_corrections} color corrections")

if debug:
    (debug_dir / "8_consensus").mkdir(exist_ok=True)
    for i, base_idx in enumerate(frame_indices):
        cv2.imwrite(str(debug_dir / "8_consensus" / f"frame_{base_idx:02d}.png"), final_frames[i])
    print(f"  Saved corrected frames to debug/8_consensus/")
```

**Step 4: Run import test**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.pipeline import ClothingPipeline
from sprite_keypoint_detector.consensus import build_consensus_map, apply_consensus_correction
print('Pipeline imports OK')
"
```

Expected: `Pipeline imports OK`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/pipeline.py
git commit -m "feat(pipeline): integrate color consensus correction"
```

---

## Task 6: Full pipeline test

**Step 1: Run pipeline with debug**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI
python3 -u -m sprite_keypoint_detector.pipeline \
    --frames-dir training_data/frames \
    --annotations training_data/annotations.json \
    --masks training_data/masks_corrected \
    --output /tmp/pipeline_consensus_test \
    --debug \
    --skip-validation
```

Expected output should include:
- `=== Applying Color Consensus Correction ===`
- `Built consensus map with N positions`
- `Applied N color corrections`

**Step 2: Verify debug output exists**

Run:
```bash
ls /tmp/pipeline_consensus_test/debug/8_consensus/ | head -5
```

Expected: `frame_00.png`, `frame_01.png`, etc.

**Step 3: View comparison strip**

Run:
```bash
open /tmp/pipeline_consensus_test/debug/comparison/frame_00.png
```

Verify: Should show new "Consensus" column after "Consistency" column.

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Create consensus.py with canonical coordinate functions |
| 2 | Add segment pixel collection with grid positions |
| 3 | Add consensus map building with plurality voting |
| 4 | Add consensus correction application |
| 5 | Integrate into pipeline |
| 6 | Full pipeline test |
