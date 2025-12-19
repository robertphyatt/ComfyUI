# Silhouette Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add post-rotation silhouette refinement that translates limb segments to minimize red pixels (armor extending beyond base silhouette).

**Architecture:** After rotation aligns bone angles, iterate through limb chains hierarchically (parent first, then children). For each segment, use gradient descent to find XY offset (±15px, 1px steps) that minimizes red pixels in that segment's armor region. Child segments cascade with parent translations, then get their own fine-tuning pass. Iterate until no improvement.

**Tech Stack:** Python, numpy, OpenCV

---

### Task 1: Add helper function to count red pixels in a region

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:278-280`

**Step 1: Add count_red_in_region function after apply_rotation**

Find (line 278-280):
```python
    return result, result_kpts


# ============ Step 3: Soft-Edge Inpaint ============
```

Insert between `return result, result_kpts` and the comment:
```python
    return result, result_kpts


# ============ Step 2.5: Silhouette Refinement ============

def _count_red_in_region(
    armor: np.ndarray,
    base_image: np.ndarray,
    region_mask: np.ndarray
) -> int:
    """Count red pixels (armor outside base) within a region.

    Args:
        armor: Armor RGBA image
        base_image: Base frame RGBA image
        region_mask: Boolean mask defining region of interest

    Returns:
        Count of pixels where armor is visible but base is not, within region
    """
    armor_visible = armor[:, :, 3] > 128
    base_visible = base_image[:, :, 3] > 128

    # Red = armor visible AND base NOT visible (armor extending beyond base)
    red_pixels = armor_visible & ~base_visible & region_mask

    return int(np.sum(red_pixels))


# ============ Step 3: Soft-Edge Inpaint ============
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import _count_red_in_region; print('OK')"`

Expected: `OK`

---

### Task 2: Add function to get armor segment mask

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py` (after _count_red_in_region)

**Step 1: Add _get_armor_segment_mask function**

Insert after `_count_red_in_region`:
```python
def _get_armor_segment_mask(
    armor: np.ndarray,
    keypoints: np.ndarray,
    joint_idx: int,
    child_idx: int,
    segment_width: int = 35
) -> np.ndarray:
    """Get mask of armor pixels belonging to a limb segment.

    Args:
        armor: Armor RGBA image
        keypoints: Current keypoints array
        joint_idx: Index of parent joint
        child_idx: Index of child joint
        segment_width: Width of segment region

    Returns:
        Boolean mask where armor pixels are within segment region
    """
    h, w = armor.shape[:2]

    # Get segment region (same as rotation uses)
    segment_region = _create_segment_mask(
        h, w, keypoints[joint_idx], keypoints[child_idx], segment_width
    )

    # Intersect with actual armor pixels
    armor_visible = armor[:, :, 3] > 128

    return armor_visible & segment_region
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import _get_armor_segment_mask; print('OK')"`

Expected: `OK`

---

### Task 3: Add function to translate armor segment

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py` (after _get_armor_segment_mask)

**Step 1: Add _translate_segment function**

Insert after `_get_armor_segment_mask`:
```python
def _translate_segment(
    armor: np.ndarray,
    keypoints: np.ndarray,
    segment_mask: np.ndarray,
    offset: Tuple[int, int],
    descendant_masks: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate armor pixels in segment and all descendants by offset.

    Args:
        armor: Armor RGBA image
        keypoints: Current keypoints array (modified in place)
        segment_mask: Boolean mask of segment pixels to move
        offset: (dx, dy) translation offset
        descendant_masks: List of masks for descendant segments (also moved)

    Returns:
        (translated_armor, updated_keypoints)
    """
    dx, dy = offset
    if dx == 0 and dy == 0:
        return armor, keypoints

    h, w = armor.shape[:2]
    result = armor.copy()
    result_kpts = keypoints.copy()

    # Combine segment mask with all descendant masks
    combined_mask = segment_mask.copy()
    for desc_mask in descendant_masks:
        combined_mask = combined_mask | desc_mask

    # Extract pixels to move
    pixels_to_move = np.zeros_like(armor)
    for c in range(4):
        pixels_to_move[:, :, c] = np.where(combined_mask, armor[:, :, c], 0)

    # Clear original positions
    for c in range(4):
        result[:, :, c] = np.where(combined_mask, 0, result[:, :, c])

    # Translate using warpAffine
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(
        pixels_to_move, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Composite translated pixels back
    trans_alpha = translated[:, :, 3:4] / 255.0
    for c in range(3):
        result[:, :, c] = (translated[:, :, c] * trans_alpha[:, :, 0] +
                          result[:, :, c] * (1 - trans_alpha[:, :, 0])).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], translated[:, :, 3])

    return result, result_kpts
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import _translate_segment; print('OK')"`

Expected: `OK`

---

### Task 4: Add gradient descent function to find optimal offset

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py` (after _translate_segment)

**Step 1: Add _find_optimal_offset function**

Insert after `_translate_segment`:
```python
def _find_optimal_offset(
    armor: np.ndarray,
    base_image: np.ndarray,
    segment_mask: np.ndarray,
    descendant_masks: List[np.ndarray],
    max_radius: int = 15
) -> Tuple[int, int]:
    """Find XY offset that minimizes red pixels in segment region.

    Uses gradient descent with 1px steps, checking 8 directions.

    Args:
        armor: Armor RGBA image
        base_image: Base frame RGBA image
        segment_mask: Boolean mask of segment's armor pixels
        descendant_masks: Masks of descendant segments (moved together)
        max_radius: Maximum offset in any direction

    Returns:
        (dx, dy) optimal offset
    """
    # Combine masks for red pixel counting
    combined_mask = segment_mask.copy()
    for desc_mask in descendant_masks:
        combined_mask = combined_mask | desc_mask

    h, w = armor.shape[:2]

    # Precompute base visibility
    base_visible = base_image[:, :, 3] > 128

    def count_red_at_offset(dx: int, dy: int) -> int:
        """Count red pixels if segment were translated by (dx, dy)."""
        # Translate the combined mask
        if dx == 0 and dy == 0:
            shifted_mask = combined_mask
        else:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_mask = cv2.warpAffine(
                combined_mask.astype(np.uint8), M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ).astype(bool)

        # Red = shifted armor region outside base
        red = shifted_mask & ~base_visible
        return int(np.sum(red))

    # Start at origin
    current_offset = (0, 0)
    best_offset = (0, 0)
    best_red = count_red_at_offset(0, 0)

    # 8 directions: cardinal + diagonal
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Gradient descent
    while True:
        improved = False
        for dx, dy in directions:
            test_x = current_offset[0] + dx
            test_y = current_offset[1] + dy

            # Check bounds
            if abs(test_x) > max_radius or abs(test_y) > max_radius:
                continue

            red = count_red_at_offset(test_x, test_y)
            if red < best_red:
                best_red = red
                best_offset = (test_x, test_y)
                current_offset = (test_x, test_y)
                improved = True
                break  # Greedy: take first improvement

        if not improved:
            break  # Local minimum reached

    return best_offset
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import _find_optimal_offset; print('OK')"`

Expected: `OK`

**Step 3: Commit helper functions**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): add silhouette refinement helper functions

- _count_red_in_region: count armor pixels outside base silhouette
- _get_armor_segment_mask: get armor pixels in limb segment
- _translate_segment: translate segment and descendants by offset
- _find_optimal_offset: gradient descent to minimize red pixels

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Add main refine_silhouette_alignment function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py` (after _find_optimal_offset)

**Step 1: Add refine_silhouette_alignment function**

Insert after `_find_optimal_offset`:
```python
def refine_silhouette_alignment(
    armor: np.ndarray,
    armor_kpts: np.ndarray,
    base_image: np.ndarray,
    base_kpts: np.ndarray,
    config: TransformConfig,
    max_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively refine armor position to minimize red pixels.

    After rotation aligns bone angles, this step translates limb segments
    to better align armor silhouette with base silhouette.

    Args:
        armor: Rotated armor RGBA image
        armor_kpts: Armor keypoints after rotation
        base_image: Base frame RGBA image
        base_kpts: Base frame keypoints
        config: Transform configuration
        max_iterations: Maximum refinement passes

    Returns:
        (refined_armor, refined_keypoints)
    """
    result = armor.copy()
    result_kpts = armor_kpts.copy()

    # Count initial red pixels for convergence check
    armor_visible = result[:, :, 3] > 128
    base_visible = base_image[:, :, 3] > 128
    prev_total_red = int(np.sum(armor_visible & ~base_visible))

    for iteration in range(max_iterations):
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

                # Find optimal offset for this segment
                offset = _find_optimal_offset(
                    result, base_image, segment_mask, descendant_masks
                )

                if offset != (0, 0):
                    # Apply translation to segment and descendants
                    result, result_kpts = _translate_segment(
                        result, result_kpts, segment_mask, offset, descendant_masks
                    )

                    # Update keypoints for this segment and descendants
                    dx, dy = offset
                    result_kpts[child_idx] = result_kpts[child_idx] + np.array([dx, dy])
                    for j in range(i + 1, len(chain)):
                        desc_joint, desc_child, _ = chain[j]
                        result_kpts[desc_child] = result_kpts[desc_child] + np.array([dx, dy])

        # Check convergence
        armor_visible = result[:, :, 3] > 128
        total_red = int(np.sum(armor_visible & ~base_visible))
        if total_red >= prev_total_red:
            break  # No improvement, stop
        prev_total_red = total_red

    return result, result_kpts
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import refine_silhouette_alignment; print('OK')"`

Expected: `OK`

---

### Task 6: Add refined_armor field to TransformDebugOutput

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:502-514`

**Step 1: Add refined_armor and refined_kpts fields to dataclass**

Find (lines 502-514):
```python
@dataclass
class TransformDebugOutput:
    """Debug outputs from transform pipeline."""
    aligned_clothed: np.ndarray       # After scale + align
    aligned_kpts: np.ndarray          # Keypoints after scale + align
    armor_masked: np.ndarray          # After applying mask
    rotated_armor: np.ndarray         # After rotation
    rotated_kpts: np.ndarray          # Keypoints after rotation
    inpainted_armor: np.ndarray       # After inpaint
    final_armor: np.ndarray           # After inpaint (pre-pixelize)
    pre_inpaint_overlap_viz: np.ndarray  # Overlap viz BEFORE inpaint (shows what needs filling)
    overlap_viz: np.ndarray           # Blue/red/green overlap visualization (after inpaint)
    skeleton_viz: np.ndarray          # Skeleton overlay
```

Replace with:
```python
@dataclass
class TransformDebugOutput:
    """Debug outputs from transform pipeline."""
    aligned_clothed: np.ndarray       # After scale + align
    aligned_kpts: np.ndarray          # Keypoints after scale + align
    armor_masked: np.ndarray          # After applying mask
    rotated_armor: np.ndarray         # After rotation
    rotated_kpts: np.ndarray          # Keypoints after rotation
    refined_armor: np.ndarray         # After silhouette refinement
    refined_kpts: np.ndarray          # Keypoints after refinement
    inpainted_armor: np.ndarray       # After inpaint
    final_armor: np.ndarray           # After inpaint (pre-pixelize)
    pre_inpaint_overlap_viz: np.ndarray  # Overlap viz BEFORE inpaint (shows what needs filling)
    post_refine_overlap_viz: np.ndarray  # Overlap viz AFTER refinement (shows improvement)
    overlap_viz: np.ndarray           # Blue/red/green overlap visualization (after inpaint)
    skeleton_viz: np.ndarray          # Skeleton overlay
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import TransformDebugOutput; print('OK')"`

Expected: Error (constructor expects new fields) - this is expected until we update the function

---

### Task 7: Integrate refinement into transform_frame

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:642-655`

**Step 1: Add refinement step to transform_frame**

Find (lines 642-655):
```python
    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor, aligned_kpts, base_kpts, config)

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        rotated_armor, aligned_clothed, base_image,
        rotated_kpts, base_kpts, config
    )

    # Pixelization now happens after color correction in pipeline
    return inpainted_armor
```

Replace with:
```python
    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor, aligned_kpts, base_kpts, config)

    # Step 2.5: Silhouette refinement
    refined_armor, refined_kpts = refine_silhouette_alignment(
        rotated_armor, rotated_kpts, base_image, base_kpts, config
    )

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        refined_armor, aligned_clothed, base_image,
        refined_kpts, base_kpts, config
    )

    # Pixelization now happens after color correction in pipeline
    return inpainted_armor
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import transform_frame; print('OK')"`

Expected: `OK`

---

### Task 8: Integrate refinement into transform_frame_debug

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:694-734`

**Step 1: Add refinement step and update debug output**

Find (lines 694-734, approximately):
```python
    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor_masked, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor_masked, aligned_kpts, base_kpts, config)

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        rotated_armor, aligned_clothed, base_image,
        rotated_kpts, base_kpts, config
    )

    # Pixelization now happens after color correction in pipeline
    final_armor = inpainted_armor

    # Create visualizations
    neck_y = int(base_kpts[1, 1])

    # Pre-inpaint overlap shows what needs to be filled (before inpainting)
    pre_inpaint_overlap_viz = _create_overlap_visualization(base_image, rotated_armor, neck_y)

    # Post-inpaint overlap shows final coverage (after inpainting)
    overlap_viz = _create_overlap_visualization(base_image, final_armor, neck_y)

    # Skeleton visualization: base skeleton (green) + armor skeleton (red) on base image
    skeleton_viz = base_image[:, :, :3].copy()
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, base_kpts, color=(0, 255, 0), thickness=2)
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, rotated_kpts, color=(0, 0, 255), thickness=1)

    return TransformDebugOutput(
        aligned_clothed=aligned_clothed,
        aligned_kpts=aligned_kpts,
        armor_masked=armor_masked,
        rotated_armor=rotated_armor,
        rotated_kpts=rotated_kpts,
        inpainted_armor=inpainted_armor,
        final_armor=final_armor,
        pre_inpaint_overlap_viz=pre_inpaint_overlap_viz,
        overlap_viz=overlap_viz,
        skeleton_viz=skeleton_viz
    )
```

Replace with:
```python
    # Step 2: Rotate (skip if fit is already good)
    if config.skip_rotation:
        rotated_armor, rotated_kpts = armor_masked, aligned_kpts
    else:
        rotated_armor, rotated_kpts = apply_rotation(armor_masked, aligned_kpts, base_kpts, config)

    # Step 2.5: Silhouette refinement
    refined_armor, refined_kpts = refine_silhouette_alignment(
        rotated_armor, rotated_kpts, base_image, base_kpts, config
    )

    # Step 3: Inpaint
    inpainted_armor = apply_inpaint(
        refined_armor, aligned_clothed, base_image,
        refined_kpts, base_kpts, config
    )

    # Pixelization now happens after color correction in pipeline
    final_armor = inpainted_armor

    # Create visualizations
    neck_y = int(base_kpts[1, 1])

    # Pre-inpaint overlap shows what needs to be filled (before refinement)
    pre_inpaint_overlap_viz = _create_overlap_visualization(base_image, rotated_armor, neck_y)

    # Post-refinement overlap shows improvement from silhouette alignment
    post_refine_overlap_viz = _create_overlap_visualization(base_image, refined_armor, neck_y)

    # Post-inpaint overlap shows final coverage (after inpainting)
    overlap_viz = _create_overlap_visualization(base_image, final_armor, neck_y)

    # Skeleton visualization: base skeleton (green) + armor skeleton (red) on base image
    skeleton_viz = base_image[:, :, :3].copy()
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, base_kpts, color=(0, 255, 0), thickness=2)
    skeleton_viz = _draw_skeleton_on_image(skeleton_viz, refined_kpts, color=(0, 0, 255), thickness=1)

    return TransformDebugOutput(
        aligned_clothed=aligned_clothed,
        aligned_kpts=aligned_kpts,
        armor_masked=armor_masked,
        rotated_armor=rotated_armor,
        rotated_kpts=rotated_kpts,
        refined_armor=refined_armor,
        refined_kpts=refined_kpts,
        inpainted_armor=inpainted_armor,
        final_armor=final_armor,
        pre_inpaint_overlap_viz=pre_inpaint_overlap_viz,
        post_refine_overlap_viz=post_refine_overlap_viz,
        overlap_viz=overlap_viz,
        skeleton_viz=skeleton_viz
    )
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import transform_frame_debug; print('OK')"`

Expected: `OK`

**Step 3: Commit transform changes**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): integrate silhouette refinement step

Adds refine_silhouette_alignment() between rotation and inpainting.
Uses gradient descent to translate limb segments, minimizing red pixels.

- Hierarchical processing: parent segments first, children cascade
- 1px steps, ±15px max radius, iterate until no improvement
- New debug outputs: refined_armor, post_refine_overlap_viz

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Update pipeline debug directories and file saving

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py:340-351`
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py:413-419`

**Step 1: Add refined debug directories**

Find (lines 340-351):
```python
        debug_dir = None
        if debug:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            (debug_dir / "1_aligned").mkdir(exist_ok=True)
            (debug_dir / "2_masked").mkdir(exist_ok=True)
            (debug_dir / "3_rotated").mkdir(exist_ok=True)
            (debug_dir / "4_inpainted").mkdir(exist_ok=True)
            (debug_dir / "6_final").mkdir(exist_ok=True)
            (debug_dir / "pre_inpaint_overlap").mkdir(exist_ok=True)
            (debug_dir / "overlap").mkdir(exist_ok=True)
            (debug_dir / "skeleton").mkdir(exist_ok=True)
```

Replace with:
```python
        debug_dir = None
        if debug:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            (debug_dir / "1_aligned").mkdir(exist_ok=True)
            (debug_dir / "2_masked").mkdir(exist_ok=True)
            (debug_dir / "3_rotated").mkdir(exist_ok=True)
            (debug_dir / "pre_inpaint_overlap").mkdir(exist_ok=True)
            (debug_dir / "4_refined").mkdir(exist_ok=True)
            (debug_dir / "post_refine_overlap").mkdir(exist_ok=True)
            (debug_dir / "5_inpainted").mkdir(exist_ok=True)
            (debug_dir / "6_final").mkdir(exist_ok=True)
            (debug_dir / "overlap").mkdir(exist_ok=True)
            (debug_dir / "skeleton").mkdir(exist_ok=True)
```

**Step 2: Update file saving**

Find (around lines 413-419):
```python
                cv2.imwrite(str(debug_dir / "1_aligned" / f"frame_{base_idx:02d}.png"), debug_output.aligned_clothed)
                cv2.imwrite(str(debug_dir / "2_masked" / f"frame_{base_idx:02d}.png"), debug_output.armor_masked)
                cv2.imwrite(str(debug_dir / "3_rotated" / f"frame_{base_idx:02d}.png"), debug_output.rotated_armor)
                cv2.imwrite(str(debug_dir / "4_inpainted" / f"frame_{base_idx:02d}.png"), debug_output.inpainted_armor)
                cv2.imwrite(str(debug_dir / "pre_inpaint_overlap" / f"frame_{base_idx:02d}.png"), debug_output.pre_inpaint_overlap_viz)
                cv2.imwrite(str(debug_dir / "overlap" / f"frame_{base_idx:02d}.png"), debug_output.overlap_viz)
                cv2.imwrite(str(debug_dir / "skeleton" / f"frame_{base_idx:02d}.png"), debug_output.skeleton_viz)
```

Replace with:
```python
                cv2.imwrite(str(debug_dir / "1_aligned" / f"frame_{base_idx:02d}.png"), debug_output.aligned_clothed)
                cv2.imwrite(str(debug_dir / "2_masked" / f"frame_{base_idx:02d}.png"), debug_output.armor_masked)
                cv2.imwrite(str(debug_dir / "3_rotated" / f"frame_{base_idx:02d}.png"), debug_output.rotated_armor)
                cv2.imwrite(str(debug_dir / "pre_inpaint_overlap" / f"frame_{base_idx:02d}.png"), debug_output.pre_inpaint_overlap_viz)
                cv2.imwrite(str(debug_dir / "4_refined" / f"frame_{base_idx:02d}.png"), debug_output.refined_armor)
                cv2.imwrite(str(debug_dir / "post_refine_overlap" / f"frame_{base_idx:02d}.png"), debug_output.post_refine_overlap_viz)
                cv2.imwrite(str(debug_dir / "5_inpainted" / f"frame_{base_idx:02d}.png"), debug_output.inpainted_armor)
                cv2.imwrite(str(debug_dir / "overlap" / f"frame_{base_idx:02d}.png"), debug_output.overlap_viz)
                cv2.imwrite(str(debug_dir / "skeleton" / f"frame_{base_idx:02d}.png"), debug_output.skeleton_viz)
```

**Step 3: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import ClothingPipeline; print('OK')"`

Expected: `OK`

---

### Task 10: Update comparison image steps

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py:59-68`

**Step 1: Update steps list**

Find (lines 59-68):
```python
    steps = [
        ("1_aligned", "Aligned"),
        ("2_masked", "Masked"),
        ("3_rotated", "Rotated"),
        ("pre_inpaint_overlap", "Pre-Inpaint"),
        ("4_inpainted", "Inpainted"),
        ("6_final", "Final"),
        ("overlap", "Overlap"),
        ("skeleton", "Skeleton"),
    ]
```

Replace with:
```python
    steps = [
        ("1_aligned", "Aligned"),
        ("2_masked", "Masked"),
        ("3_rotated", "Rotated"),
        ("pre_inpaint_overlap", "Pre-Refine"),
        ("4_refined", "Refined"),
        ("post_refine_overlap", "Post-Refine"),
        ("5_inpainted", "Inpainted"),
        ("6_final", "Final"),
        ("overlap", "Overlap"),
        ("skeleton", "Skeleton"),
    ]
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import create_debug_comparison; print('OK')"`

Expected: `OK`

**Step 3: Commit pipeline changes**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): add refined debug output and comparison steps

Updates debug output to show:
- 4_refined/: armor after silhouette refinement
- post_refine_overlap/: overlap viz after refinement
- Renamed 4_inpainted -> 5_inpainted for proper ordering

Comparison now shows: Aligned -> Masked -> Rotated -> Pre-Refine ->
Refined -> Post-Refine -> Inpainted -> Final -> Overlap -> Skeleton

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 11: Test full pipeline

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

Expected: Pipeline completes without errors

**Step 2: Verify new debug folders exist**

```bash
ls /tmp/pipeline_test/debug/4_refined/
ls /tmp/pipeline_test/debug/post_refine_overlap/
```

Expected: `frame_00.png`, `frame_01.png`, ... files in each

**Step 3: Check comparison image includes new steps**

```bash
open /tmp/pipeline_test/debug/comparison/frame_00.png
```

Expected: Comparison now has 10 columns including "Refined" and "Post-Refine"

**Step 4: Compare pre-refine vs post-refine overlap**

Visually compare `pre_inpaint_overlap/frame_00.png` vs `post_refine_overlap/frame_00.png`:
- Post-refine should have LESS red pixels (armor outside base)
- The red/blue balance on the right arm should be improved

---

## Summary

This plan adds silhouette refinement between rotation and inpainting:

1. **Helper functions** (Tasks 1-4): `_count_red_in_region`, `_get_armor_segment_mask`, `_translate_segment`, `_find_optimal_offset`
2. **Main function** (Task 5): `refine_silhouette_alignment()` - iterative hierarchical gradient descent
3. **Debug output** (Tasks 6-10): New fields, folders, and comparison steps
4. **Testing** (Task 11): Verify pipeline runs and produces expected debug output

The refinement step:
- Processes limb chains hierarchically (parent first, children cascade)
- Uses gradient descent (1px steps, ±15px max) to minimize red pixels
- Iterates until no improvement (max 50 iterations)
- Produces debug visualizations showing before/after refinement
