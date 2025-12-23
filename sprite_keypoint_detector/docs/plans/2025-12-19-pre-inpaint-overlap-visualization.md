# Pre-Inpaint Overlap Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a blue/red/green overlap visualization showing armor coverage AFTER rotation but BEFORE inpainting, to help debug alignment issues.

**Architecture:** Add a new field `pre_inpaint_overlap_viz` to `TransformDebugOutput`, create the visualization using the existing `_create_overlap_visualization()` function on `rotated_armor`, save it to a new debug folder, and include it in the comparison images.

**Tech Stack:** Python, numpy, OpenCV

---

### Task 1: Add pre_inpaint_overlap_viz field to TransformDebugOutput

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:503-513`

**Step 1: Add new field to dataclass**

Find (lines 503-513):
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
    overlap_viz: np.ndarray           # Blue/red/green overlap visualization
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
    inpainted_armor: np.ndarray       # After inpaint
    final_armor: np.ndarray           # After inpaint (pre-pixelize)
    pre_inpaint_overlap_viz: np.ndarray  # Overlap viz BEFORE inpaint (shows what needs filling)
    overlap_viz: np.ndarray           # Blue/red/green overlap visualization (after inpaint)
    skeleton_viz: np.ndarray          # Skeleton overlay
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import TransformDebugOutput; print('OK')"`

Expected: Error (constructor now expects new field)

---

### Task 2: Create pre_inpaint_overlap_viz in transform_frame_debug

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py:705-727`

**Step 1: Add pre-inpaint overlap visualization creation**

Find (around lines 705-727):
```python
    # Pixelization now happens after color correction in pipeline
    final_armor = inpainted_armor

    # Create visualizations
    neck_y = int(base_kpts[1, 1])
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
        overlap_viz=overlap_viz,
        skeleton_viz=skeleton_viz
    )
```

Replace with:
```python
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

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import transform_frame_debug; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): add pre-inpaint overlap visualization

Shows blue/red/green overlap BEFORE inpainting to help debug
what areas need to be filled vs what's already covered.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Add pre_inpaint_overlap directory and save visualization

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py:340-350` and `410-416`

**Step 1: Create debug directory**

Find (around lines 340-350):
```python
        if debug:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            (debug_dir / "1_aligned").mkdir(exist_ok=True)
            (debug_dir / "2_masked").mkdir(exist_ok=True)
            (debug_dir / "3_rotated").mkdir(exist_ok=True)
            (debug_dir / "4_inpainted").mkdir(exist_ok=True)
            (debug_dir / "6_final").mkdir(exist_ok=True)
            (debug_dir / "overlap").mkdir(exist_ok=True)
            (debug_dir / "skeleton").mkdir(exist_ok=True)
```

Replace with:
```python
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

**Step 2: Save pre-inpaint overlap image**

Find (around lines 410-416):
```python
                cv2.imwrite(str(debug_dir / "1_aligned" / f"frame_{base_idx:02d}.png"), debug_output.aligned_clothed)
                cv2.imwrite(str(debug_dir / "2_masked" / f"frame_{base_idx:02d}.png"), debug_output.armor_masked)
                cv2.imwrite(str(debug_dir / "3_rotated" / f"frame_{base_idx:02d}.png"), debug_output.rotated_armor)
                cv2.imwrite(str(debug_dir / "4_inpainted" / f"frame_{base_idx:02d}.png"), debug_output.inpainted_armor)
                cv2.imwrite(str(debug_dir / "overlap" / f"frame_{base_idx:02d}.png"), debug_output.overlap_viz)
                cv2.imwrite(str(debug_dir / "skeleton" / f"frame_{base_idx:02d}.png"), debug_output.skeleton_viz)
```

Replace with:
```python
                cv2.imwrite(str(debug_dir / "1_aligned" / f"frame_{base_idx:02d}.png"), debug_output.aligned_clothed)
                cv2.imwrite(str(debug_dir / "2_masked" / f"frame_{base_idx:02d}.png"), debug_output.armor_masked)
                cv2.imwrite(str(debug_dir / "3_rotated" / f"frame_{base_idx:02d}.png"), debug_output.rotated_armor)
                cv2.imwrite(str(debug_dir / "4_inpainted" / f"frame_{base_idx:02d}.png"), debug_output.inpainted_armor)
                cv2.imwrite(str(debug_dir / "pre_inpaint_overlap" / f"frame_{base_idx:02d}.png"), debug_output.pre_inpaint_overlap_viz)
                cv2.imwrite(str(debug_dir / "overlap" / f"frame_{base_idx:02d}.png"), debug_output.overlap_viz)
                cv2.imwrite(str(debug_dir / "skeleton" / f"frame_{base_idx:02d}.png"), debug_output.skeleton_viz)
```

**Step 3: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import ClothingPipeline; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): save pre-inpaint overlap visualization to debug

Adds pre_inpaint_overlap/ debug folder showing armor coverage
before inpainting fills the gaps.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Add pre-inpaint overlap to comparison images

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py:59-67`

**Step 1: Update steps list**

Find (around lines 59-67):
```python
    steps = [
        ("1_aligned", "Aligned"),
        ("2_masked", "Masked"),
        ("3_rotated", "Rotated"),
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
        ("pre_inpaint_overlap", "Pre-Inpaint"),
        ("4_inpainted", "Inpainted"),
        ("6_final", "Final"),
        ("overlap", "Overlap"),
        ("skeleton", "Skeleton"),
    ]
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import create_debug_comparison; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): add pre-inpaint overlap to comparison images

Comparison now shows: Aligned -> Masked -> Rotated -> Pre-Inpaint ->
Inpainted -> Final -> Overlap -> Skeleton

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Test full pipeline

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

**Step 2: Verify new debug folder exists**

```bash
ls /tmp/pipeline_test/debug/pre_inpaint_overlap/
```

Expected: `frame_00.png`, `frame_01.png`, ... files

**Step 3: Check comparison image includes new step**

```bash
open /tmp/pipeline_test/debug/comparison/frame_00.png
```

Expected: Comparison now has 8 columns including "Pre-Inpaint" showing blue/red/green overlap BEFORE inpainting

---

## Summary

This plan adds:
1. New `pre_inpaint_overlap_viz` field to `TransformDebugOutput`
2. Creation of pre-inpaint overlap using `rotated_armor` (before inpainting)
3. New `pre_inpaint_overlap/` debug directory
4. "Pre-Inpaint" column in comparison images between "Rotated" and "Inpainted"

The visualization shows:
- **Green**: Base character covered by armor (good)
- **Blue**: Base character NOT covered by armor (needs inpainting)
- **Red**: Armor extending beyond base (floating armor)
