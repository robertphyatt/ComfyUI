# Center-of-Mass Constraint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure clothing sprite center-of-mass moves by exactly the same delta as base body center-of-mass, eliminating frame-to-frame jiggle from keypoint drift.

**Architecture:** After initial keypoint-based alignment and rotation (but before refinement), compute actual pixel centroids and shift clothing to match expected position based on base body movement. This corrects drift that keypoint alignment misses.

**Tech Stack:** NumPy, OpenCV, existing transform.py and pipeline.py infrastructure

---

## Background

**Problem:** Frames 16-17 show jiggle where clothing moves more/less than base body:
```
Frame 15→16: Base (-0.9, -2.3), Clothing (-0.9, -3.9) → DRIFT (+0.0, -1.7)
Frame 17→18: Base (-2.3, +3.8), Clothing (-0.8, +1.7) → DRIFT (+1.6, -2.0)
```

**Root cause:** Keypoint-based alignment doesn't perfectly track visual center-of-mass. Different clothed sources have pose variations that don't fully normalize.

**Solution:** Add pixel-based centroid constraint after alignment, before refinement.

---

### Task 1: Add compute_centroid() helper function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py` (add after imports, ~line 30)

**Step 1: Add the helper function**

Add this function after the imports section:

```python
def compute_centroid(image: np.ndarray) -> Optional[Tuple[float, float]]:
    """Compute centroid of non-transparent pixels.

    Args:
        image: RGBA image array

    Returns:
        (x, y) centroid or None if no visible pixels
    """
    if image.shape[2] < 4:
        return None
    alpha = image[:, :, 3]
    y_coords, x_coords = np.where(alpha > 128)
    if len(x_coords) == 0:
        return None
    return (float(np.mean(x_coords)), float(np.mean(y_coords)))
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from transform import compute_centroid; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/transform.py
git commit -m "feat: add compute_centroid() helper for CoM constraint"
```

---

### Task 2: Add apply_com_constraint() function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py` (add after compute_centroid)

**Step 1: Add the constraint function**

```python
def apply_com_constraint(
    armor: np.ndarray,
    base_image: np.ndarray,
    anchor_base_com: Optional[Tuple[float, float]],
    anchor_armor_com: Optional[Tuple[float, float]]
) -> np.ndarray:
    """Shift armor so its center-of-mass delta matches base body's delta.

    This corrects drift that keypoint-based alignment misses by ensuring
    the armor sprite moves by exactly the same amount as the base body.

    Args:
        armor: Current armor image (RGBA)
        base_image: Current base body image (RGBA)
        anchor_base_com: Base body centroid from frame 0 (None to skip)
        anchor_armor_com: Armor centroid from frame 0 after alignment (None to skip)

    Returns:
        Shifted armor image
    """
    # Skip if no anchor data (frame 0)
    if anchor_base_com is None or anchor_armor_com is None:
        return armor

    # Compute current centroids
    current_base_com = compute_centroid(base_image)
    current_armor_com = compute_centroid(armor)

    if current_base_com is None or current_armor_com is None:
        return armor

    # How much did base body move from frame 0?
    base_delta_x = current_base_com[0] - anchor_base_com[0]
    base_delta_y = current_base_com[1] - anchor_base_com[1]

    # Where should armor centroid be?
    expected_armor_x = anchor_armor_com[0] + base_delta_x
    expected_armor_y = anchor_armor_com[1] + base_delta_y

    # How much do we need to shift armor?
    shift_x = int(round(expected_armor_x - current_armor_com[0]))
    shift_y = int(round(expected_armor_y - current_armor_com[1]))

    # Skip if no shift needed
    if shift_x == 0 and shift_y == 0:
        return armor

    # Apply shift using translation matrix
    h, w = armor.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(armor, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return shifted
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from transform import apply_com_constraint; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/transform.py
git commit -m "feat: add apply_com_constraint() to eliminate keypoint drift"
```

---

### Task 3: Update transform_frame() signature and add CoM step

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:870-942`

**Step 1: Update function signature**

Change the signature (~line 870) from:
```python
def transform_frame(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None,
    anchor_offset: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
```

To:
```python
def transform_frame(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None,
    anchor_offset: Optional[Tuple[int, int]] = None,
    anchor_base_com: Optional[Tuple[float, float]] = None,
    anchor_armor_com: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Tuple[int, int], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Run full transform pipeline on a single frame.

    ...existing docstring...

    Returns:
        (final_armor, offset_used, base_com, armor_com_after_rotation)
        - base_com and armor_com are returned for frame 0 to establish anchors
    """
```

**Step 2: Add CoM constraint after rotation, before refinement**

After the rotation step (~line 920) and before refinement (~line 922), add:

```python
    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor, aligned_kpts, base_kpts, config)

    # Step 2.25: Center-of-mass constraint
    # Compute centroids for anchoring (frame 0) or correction (subsequent frames)
    current_base_com = compute_centroid(base_image)
    current_armor_com = compute_centroid(rotated_armor)

    # Apply CoM constraint to eliminate keypoint drift
    rotated_armor = apply_com_constraint(
        rotated_armor, base_image,
        anchor_base_com, anchor_armor_com
    )

    # Step 2.5: Silhouette refinement
    # ...existing code...
```

**Step 3: Update return statement**

Change the return at end of function (~line 941) from:
```python
    return inpainted_armor, offset_used
```

To:
```python
    return inpainted_armor, offset_used, current_base_com, current_armor_com
```

**Step 4: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from transform import transform_frame; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/transform.py
git commit -m "feat: integrate CoM constraint into transform_frame()"
```

---

### Task 4: Update transform_frame_debug() with same changes

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:944-1043`

**Step 1: Update signature**

Change signature (~line 944) to match transform_frame:
```python
def transform_frame_debug(
    clothed_image: np.ndarray,
    clothed_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    armor_mask: np.ndarray,
    config: Optional[TransformConfig] = None,
    anchor_offset: Optional[Tuple[int, int]] = None,
    anchor_base_com: Optional[Tuple[float, float]] = None,
    anchor_armor_com: Optional[Tuple[float, float]] = None
) -> Tuple[TransformDebugOutput, Tuple[int, int], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
```

**Step 2: Add CoM constraint after rotation**

After rotation (~line 988) and before refinement (~line 990), add:
```python
    # Step 2.25: Center-of-mass constraint
    current_base_com = compute_centroid(base_image)
    current_armor_com = compute_centroid(rotated_armor)

    rotated_armor = apply_com_constraint(
        rotated_armor, base_image,
        anchor_base_com, anchor_armor_com
    )
```

**Step 3: Update return statement**

Change return (~line 1042) from:
```python
    ), offset_used
```

To:
```python
    ), offset_used, current_base_com, current_armor_com
```

**Step 4: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from transform import transform_frame_debug; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/transform.py
git commit -m "feat: integrate CoM constraint into transform_frame_debug()"
```

---

### Task 5: Update pipeline.py to track and pass CoM anchors

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py:380-515`

**Step 1: Add anchor variables**

After the existing anchor variables (~line 390), add:
```python
        # Track frame 0's centroid data for center-of-mass constraint
        anchor_base_com = None   # Base body centroid from frame 0
        anchor_armor_com = None  # Armor centroid from frame 0 (after alignment+rotation)
```

**Step 2: Update debug transform call**

Change the debug transform call (~line 475) from:
```python
                debug_output, offset_used = transform_frame_debug(
                    clothed_frame, clothed_kpts,
                    base_frame, base_kpts,
                    mask, frame_config,
                    adjusted_offset
                )
```

To:
```python
                debug_output, offset_used, base_com, armor_com = transform_frame_debug(
                    clothed_frame, clothed_kpts,
                    base_frame, base_kpts,
                    mask, frame_config,
                    adjusted_offset,
                    anchor_base_com, anchor_armor_com
                )
```

**Step 3: Update normal transform call**

Change the normal transform call (~line 496) from:
```python
                transformed, offset_used = transform_frame(
                    clothed_frame, clothed_kpts,
                    base_frame, base_kpts,
                    mask, frame_config,
                    adjusted_offset
                )
```

To:
```python
                transformed, offset_used, base_com, armor_com = transform_frame(
                    clothed_frame, clothed_kpts,
                    base_frame, base_kpts,
                    mask, frame_config,
                    adjusted_offset,
                    anchor_base_com, anchor_armor_com
                )
```

**Step 4: Capture frame 0 CoM anchors**

Update the anchor capture block (~line 505) from:
```python
            if anchor_offset is None:
                anchor_offset = offset_used
                frame0_base_center = base_center.copy()
                anchor_clothed_center = clothed_center.copy()
                print(f"    (anchors set: offset={anchor_offset}, base={base_center}, clothed={clothed_center})")
```

To:
```python
            if anchor_offset is None:
                anchor_offset = offset_used
                frame0_base_center = base_center.copy()
                anchor_clothed_center = clothed_center.copy()
                anchor_base_com = base_com
                anchor_armor_com = armor_com
                print(f"    (anchors set: offset={anchor_offset}, base_com={base_com}, armor_com={armor_com})")
```

**Step 5: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "from pipeline import ClothingPipeline; print('OK')"`

Expected: `OK`

**Step 6: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_keypoint_detector/pipeline.py
git commit -m "feat: track and pass CoM anchors through pipeline"
```

---

### Task 6: Run pipeline and verify fix

**Files:**
- None (testing only)

**Step 1: Run pipeline with debug output**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
python3 pipeline.py \
  --base-spritesheet /Users/roberthyatt/Code/roleplaying-agents/godot/paper_doll/paper-doll/sprites/bodies/fighter-male-athletic/isometric/walk_south.png \
  --clothed-spritesheet /Users/roberthyatt/Code/roleplaying-agents/godot/paper_doll/paper-doll/sprites/clothing/leather2/fighter-male-athletic/isometric/walk_south_clothed.png \
  --output-dir /tmp/pipeline_test \
  --debug
```

Expected: Pipeline completes without errors, prints CoM anchor info for frame 0

**Step 2: Verify centroid drift is eliminated**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && python3 -c "
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')
from spritesheet import split_spritesheet, SpritesheetLayout

def compute_centroid(img_array):
    alpha = img_array[:, :, 3]
    y_coords, x_coords = np.where(alpha > 128)
    if len(x_coords) == 0:
        return None
    return (np.mean(x_coords), np.mean(y_coords))

base_path = '/Users/roberthyatt/Code/roleplaying-agents/godot/paper_doll/paper-doll/sprites/bodies/fighter-male-athletic/isometric/walk_south.png'
base_sheet = np.array(Image.open(base_path).convert('RGBA'))
layout = SpritesheetLayout(frame_width=512, frame_height=512, columns=5, rows=5, total_frames=25)
base_frames = split_spritesheet(base_sheet, layout)

output_dir = '/tmp/pipeline_test/debug/6_final'
print('=== Frame-to-Frame Drift (should be ~0) ===')
base_centroids = {i: compute_centroid(base_frames[i]) for i in [15, 16, 17, 18]}
clothing_centroids = {}
for i in [15, 16, 17, 18]:
    img = np.array(Image.open(f'{output_dir}/frame_{i:02d}.png').convert('RGBA'))
    clothing_centroids[i] = compute_centroid(img)

for i in [16, 17, 18]:
    base_dx = base_centroids[i][0] - base_centroids[i-1][0]
    base_dy = base_centroids[i][1] - base_centroids[i-1][1]
    cloth_dx = clothing_centroids[i][0] - clothing_centroids[i-1][0]
    cloth_dy = clothing_centroids[i][1] - clothing_centroids[i-1][1]
    drift_x = cloth_dx - base_dx
    drift_y = cloth_dy - base_dy
    print(f'  Frame {i-1}->{i}: drift=({drift_x:+.1f}, {drift_y:+.1f})')
"
```

Expected: Drift values should be close to 0 (within ±0.5 pixels)

**Step 3: Copy to Godot for visual testing**

```bash
cp /tmp/pipeline_test/clothing.png /Users/roberthyatt/Code/roleplaying-agents/godot/paper_doll/paper-doll/sprites/clothing/leather2/fighter-male-athletic/isometric/walk_south.png
```

**Step 4: Commit all changes**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add -A
git commit -m "feat: center-of-mass constraint eliminates frame-to-frame jiggle

- compute_centroid() calculates pixel-based center of mass
- apply_com_constraint() shifts armor to match base body movement
- Pipeline tracks anchor centroids from frame 0
- Constraint applied after rotation, before refinement

Fixes jiggle in frames 16-17 where clothing moved more than base body."
```

---

## Summary

The implementation adds a center-of-mass constraint that:

1. **Frame 0:** Records base body centroid and armor centroid as anchors
2. **Subsequent frames:**
   - Computes how much base body centroid moved from anchor
   - Computes where armor centroid should be (anchor + base_delta)
   - Shifts armor to match expected position

This runs **after rotation, before refinement** so refinement can then fine-tune limbs on a correctly-positioned base.
