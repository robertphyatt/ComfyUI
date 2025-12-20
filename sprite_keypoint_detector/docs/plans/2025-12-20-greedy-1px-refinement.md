# Greedy 1px Per-Segment Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken batch-then-revert refinement with greedy 1px per-segment moves that check global metrics immediately after each move.

**Architecture:** Each segment attempts one 1px move per round. After each move, check global red/blue - revert if either increased. Continue looping through all segments until a full round produces zero valid moves. Parent segments cascade movement to children; children can fine-tune independently afterward.

**Tech Stack:** Python, NumPy, OpenCV (existing dependencies)

---

### Task 1: Replace `_find_optimal_offset` with `_try_1px_move`

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:378-475`

**Step 1: Delete `_find_optimal_offset` function**

Remove the entire function from lines 378-475. This function tried to find the optimal offset using gradient descent on local metrics, which doesn't work.

**Step 2: Add new `_try_1px_move` function**

Add this function in its place:

```python
def _try_1px_move(
    armor: np.ndarray,
    base_image: np.ndarray,
    segment_mask: np.ndarray,
    descendant_masks: List[np.ndarray],
    prev_red: int,
    prev_blue: int
) -> Tuple[Tuple[int, int], int, int]:
    """Try 1px moves in 8 directions, return first that improves global metrics.

    Args:
        armor: Current armor RGBA image
        base_image: Base frame RGBA image
        segment_mask: Mask of pixels in this segment
        descendant_masks: Masks of descendant segments (moved together)
        prev_red: Current global red pixel count
        prev_blue: Current global blue pixel count

    Returns:
        ((dx, dy), new_red, new_blue) - the move and resulting metrics
        ((0, 0), prev_red, prev_blue) if no valid move found
    """
    base_visible = base_image[:, :, 3] > 128
    h, w = armor.shape[:2]

    # Combine segment mask with descendants
    combined_mask = segment_mask.copy()
    for desc_mask in descendant_masks:
        combined_mask = combined_mask | desc_mask

    # 8 directions: cardinal + diagonal
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        # Translate the segment
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Create translated armor
        test_armor = armor.copy()

        # Extract pixels to move
        pixels_to_move = np.zeros_like(armor)
        for c in range(4):
            pixels_to_move[:, :, c] = np.where(combined_mask, armor[:, :, c], 0)

        # Clear original positions
        for c in range(4):
            test_armor[:, :, c] = np.where(combined_mask, 0, test_armor[:, :, c])

        # Translate
        translated = cv2.warpAffine(
            pixels_to_move, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Composite back
        trans_alpha = translated[:, :, 3:4] / 255.0
        for c in range(3):
            test_armor[:, :, c] = (translated[:, :, c] * trans_alpha[:, :, 0] +
                                   test_armor[:, :, c] * (1 - trans_alpha[:, :, 0])).astype(np.uint8)
        test_armor[:, :, 3] = np.maximum(test_armor[:, :, 3], translated[:, :, 3])

        # Check global metrics
        armor_visible = test_armor[:, :, 3] > 128
        new_red = int(np.sum(armor_visible & ~base_visible))
        new_blue = int(np.sum(base_visible & ~armor_visible))

        # Accept if neither increased AND at least one improved
        if new_red <= prev_red and new_blue <= prev_blue and (new_red < prev_red or new_blue < prev_blue):
            return (dx, dy), new_red, new_blue

    # No valid move found
    return (0, 0), prev_red, prev_blue
```

**Step 3: Verify syntax**

Run: `python3 -m py_compile /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`
Expected: No output (success)

---

### Task 2: Rewrite `refine_silhouette_alignment` with greedy loop

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:478-578`

**Step 1: Replace the entire function**

Replace `refine_silhouette_alignment` with:

```python
def refine_silhouette_alignment(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    config: TransformConfig,
    max_iterations: int = 50  # Unused, kept for API compatibility
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively refine armor position using greedy 1px per-segment moves.

    Each segment attempts one 1px move per round. Moves are accepted only if
    global red AND blue don't increase (and at least one improves). Parent
    segments cascade movement to children. Loop continues until no segment
    can make a valid move.

    Args:
        armor: Rotated armor RGBA image
        armor_kpts: Armor keypoints after rotation
        base_image: Base frame RGBA image
        base_kpts: Base frame keypoints (unused, kept for API compatibility)
        config: Transform configuration
        max_iterations: Unused, kept for API compatibility

    Returns:
        (refined_armor, refined_keypoints)
    """
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    # Compute initial global metrics
    base_visible = base_image[:, :, 3] > 128
    armor_visible = result[:, :, 3] > 128
    curr_red = int(np.sum(armor_visible & ~base_visible))
    curr_blue = int(np.sum(base_visible & ~armor_visible))

    # Loop until no segment can improve
    while True:
        any_moved = False

        # Process all limb chains hierarchically
        for chain in LIMB_CHAINS:
            for i, (joint_idx, child_idx, name) in enumerate(chain):
                # Get segment mask
                segment_mask = _get_armor_segment_mask(
                    result, result_kpts, joint_idx, child_idx,
                    config.rotation_segment_width
                )

                if not np.any(segment_mask):
                    continue

                # Get descendant masks (remaining segments in chain)
                descendant_masks = []
                for j in range(i + 1, len(chain)):
                    desc_joint, desc_child, _ = chain[j]
                    desc_mask = _get_armor_segment_mask(
                        result, result_kpts, desc_joint, desc_child,
                        config.rotation_segment_width
                    )
                    if np.any(desc_mask):
                        descendant_masks.append(desc_mask)

                # Try 1px move in each direction
                offset, new_red, new_blue = _try_1px_move(
                    result, base_image, segment_mask, descendant_masks,
                    curr_red, curr_blue
                )

                if offset != (0, 0):
                    # Apply the move
                    result, result_kpts = _translate_segment(
                        result, result_kpts, segment_mask, offset, descendant_masks
                    )

                    # Update keypoints for this segment and descendants
                    dx, dy = offset
                    result_kpts[child_idx] = result_kpts[child_idx] + np.array([dx, dy])
                    for j in range(i + 1, len(chain)):
                        desc_joint, desc_child, _ = chain[j]
                        result_kpts[desc_child] = result_kpts[desc_child] + np.array([dx, dy])

                    # Update metrics
                    curr_red = new_red
                    curr_blue = new_blue
                    any_moved = True

        # If no segment moved this round, we're converged
        if not any_moved:
            break

    return result, result_kpts
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`
Expected: No output (success)

---

### Task 3: Test the new refinement

**Files:**
- None (manual test)

**Step 1: Clear pycache and run pipeline**

```bash
rm -rf /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/__pycache__
rm -rf /tmp/pipeline_test
cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline \
  --frames-dir training_data/frames \
  --annotations training_data/annotations.json \
  --masks training_data/masks_corrected \
  --output /tmp/pipeline_test \
  --debug --skip-validation
```

Expected: Pipeline completes without errors

**Step 2: Verify frame 00 metrics improved or stayed same**

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
import cv2
import numpy as np
from pathlib import Path

debug_dir = Path('/tmp/pipeline_test/debug')
frames_dir = Path('training_data/frames')

rotated = cv2.imread(str(debug_dir / '3_rotated/frame_00.png'), cv2.IMREAD_UNCHANGED)
refined = cv2.imread(str(debug_dir / '4_refined/frame_00.png'), cv2.IMREAD_UNCHANGED)
base = cv2.imread(str(frames_dir / 'base_frame_00.png'), cv2.IMREAD_UNCHANGED)

base_visible = base[:,:,3] > 128

rot_vis = rotated[:,:,3] > 128
rot_red = np.sum(rot_vis & ~base_visible)
rot_blue = np.sum(base_visible & ~rot_vis)

ref_vis = refined[:,:,3] > 128
ref_red = np.sum(ref_vis & ~base_visible)
ref_blue = np.sum(base_visible & ~ref_vis)

print(f'Rotated: red={rot_red}, blue={rot_blue}')
print(f'Refined: red={ref_red}, blue={ref_blue}')
print(f'Red change: {ref_red - rot_red:+d}')
print(f'Blue change: {ref_blue - rot_blue:+d}')

assert ref_red <= rot_red, 'Red increased!'
assert ref_blue <= rot_blue, 'Blue increased!'
print('PASS: Neither metric increased')
"
```

Expected: "PASS: Neither metric increased" with at least one metric showing improvement

**Step 3: Visual check**

```bash
open /tmp/pipeline_test/debug/comparison/frame_00.png
```

Expected: Pre-Refine and Post-Refine show visible improvement (less red/blue mismatch)

---

### Task 4: Verify all frames pass constraints

**Step 1: Check all 25 frames**

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
import cv2
import numpy as np
from pathlib import Path

debug_dir = Path('/tmp/pipeline_test/debug')
frames_dir = Path('training_data/frames')

print('Frame | Rot Red | Rot Blue | Ref Red | Ref Blue | Status')
print('-' * 65)

all_pass = True
for i in range(25):
    rotated = cv2.imread(str(debug_dir / f'3_rotated/frame_{i:02d}.png'), cv2.IMREAD_UNCHANGED)
    refined = cv2.imread(str(debug_dir / f'4_refined/frame_{i:02d}.png'), cv2.IMREAD_UNCHANGED)
    base = cv2.imread(str(frames_dir / f'base_frame_{i:02d}.png'), cv2.IMREAD_UNCHANGED)

    base_visible = base[:,:,3] > 128

    rot_vis = rotated[:,:,3] > 128
    rot_red = np.sum(rot_vis & ~base_visible)
    rot_blue = np.sum(base_visible & ~rot_vis)

    ref_vis = refined[:,:,3] > 128
    ref_red = np.sum(ref_vis & ~base_visible)
    ref_blue = np.sum(base_visible & ~ref_vis)

    if ref_red > rot_red or ref_blue > rot_blue:
        status = 'FAIL'
        all_pass = False
    elif ref_red < rot_red or ref_blue < rot_blue:
        status = 'Better'
    else:
        status = 'Same'

    print(f'{i:02d}    | {rot_red:7d} | {rot_blue:8d} | {ref_red:7d} | {ref_blue:8d} | {status}')

print()
print('ALL PASS' if all_pass else 'SOME FAILED')
"
```

Expected: "ALL PASS" - no frame shows increased red or blue

---

### Task 5: Commit the changes

**Step 1: Stage and commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && git add transform.py && git commit -m "$(cat <<'EOF'
fix(transform): replace batch refinement with greedy 1px per-segment moves

The old refinement optimized segments independently using local metrics,
then checked global metrics at the end - causing valid local moves to
combine into invalid global changes that got reverted.

New approach:
- Each segment attempts one 1px move per round
- Global red/blue checked immediately after each move
- Move accepted only if neither metric increases
- Loop until no segment can improve
- Parent segments cascade to children; children can fine-tune after

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Step 2: Push**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector && git push
```

---

## Summary

This plan replaces the broken batch-then-revert refinement with a greedy algorithm that:
1. Tries 1px moves per segment
2. Checks global metrics immediately
3. Accepts only moves that don't increase either red or blue
4. Loops until converged

The key insight is checking global metrics after each individual move rather than after all segments have moved.
