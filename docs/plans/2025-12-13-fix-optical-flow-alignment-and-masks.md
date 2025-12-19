# Fix Optical Flow Alignment Detection and Mask Transparency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two bugs in the optical flow clothing pipeline: (1) alignment detection incorrectly reports perfect alignment when mannequin shoulders peek through armor, (2) mask editor allows painting on transparent background pixels.

**Architecture:** The alignment check must use alpha channel comparison (not grayscale silhouettes) to detect when reference clothing doesn't fully cover the mannequin. The mask validation must load RGBA images and apply `remove_transparent_background()` after each edit.

**Tech Stack:** Python, PIL, NumPy, OpenCV, matplotlib

---

## Problem Analysis

### Problem 1: False Positive Alignment Detection
- Current: `images_already_aligned()` compares grayscale silhouettes via `create_body_mask()`
- Issue: Grayscale silhouettes are similar (IoU ~0.98+) even when alpha coverage differs
- Evidence: Frame 5 has 2,622 pixels where mannequin is visible but armor is not
- Result: Pipeline skips warping, mannequin shoulders show through armor

### Problem 2: Mask Editor Allows Transparent Pixels
- Current: `_validate_masks()` loads RGB images, doesn't apply `remove_transparent_background()`
- Issue: User can paint clothing mask on transparent background areas
- Solution: Load RGBA, pass alpha to `remove_transparent_background()` after edits

---

### Task 1: Add Alpha-Based Alignment Check Function

**Files:**
- Modify: `sprite_clothing_gen/optical_flow.py:148-178`
- Test: `tests/test_optical_flow.py`

**Step 1: Write the failing test**

Add to `tests/test_optical_flow.py`:

```python
def test_images_already_aligned_detects_alpha_mismatch(tmp_path):
    """Alignment check should fail when alpha channels differ significantly."""
    from sprite_clothing_gen.optical_flow import images_already_aligned_alpha
    from PIL import Image
    import numpy as np

    # Create base image with full body visible (alpha=255 for body area)
    base = Image.new('RGBA', (100, 100), (255, 255, 255, 0))  # Transparent bg
    base_arr = np.array(base)
    base_arr[20:80, 30:70, :3] = [128, 128, 128]  # Gray body
    base_arr[20:80, 30:70, 3] = 255  # Visible
    base_path = tmp_path / "base.png"
    Image.fromarray(base_arr).save(base_path)

    # Create clothed image with SMALLER visible area (shoulders missing)
    clothed = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
    clothed_arr = np.array(clothed)
    clothed_arr[30:80, 30:70, :3] = [139, 69, 19]  # Brown armor (smaller)
    clothed_arr[30:80, 30:70, 3] = 255  # Visible
    clothed_path = tmp_path / "clothed.png"
    Image.fromarray(clothed_arr).save(clothed_path)

    # Should NOT be aligned - base has visible pixels clothed doesn't cover
    result = images_already_aligned_alpha(clothed_path, base_path, threshold=0.98)
    assert result == False, "Should detect alpha mismatch (mannequin showing through)"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -m pytest tests/test_optical_flow.py::test_images_already_aligned_detects_alpha_mismatch -v`

Expected: FAIL with "cannot import name 'images_already_aligned_alpha'"

**Step 3: Write minimal implementation**

Add to `sprite_clothing_gen/optical_flow.py` after `create_body_mask_from_alpha`:

```python
def images_already_aligned_alpha(
    clothed_path: Path,
    mannequin_path: Path,
    threshold: float = 0.98
) -> bool:
    """Check if clothed image fully covers mannequin using alpha channels.

    Critical: If ANY mannequin pixels are visible that clothed doesn't cover,
    we need warping to fill those gaps. This catches shoulder peek-through issues.

    Args:
        clothed_path: Path to clothed reference frame (RGBA)
        mannequin_path: Path to base mannequin frame (RGBA)
        threshold: Coverage threshold (0-1), how much of mannequin must be covered

    Returns:
        True only if clothed covers ALL visible mannequin pixels
    """
    # Load alpha channels
    mannequin_alpha = np.array(Image.open(mannequin_path).convert('RGBA'))[:, :, 3]
    clothed_alpha = np.array(Image.open(clothed_path).convert('RGBA'))[:, :, 3]

    # Where mannequin is visible
    mannequin_visible = mannequin_alpha > 128

    # Where clothed is visible
    clothed_visible = clothed_alpha > 128

    # Critical check: mannequin pixels that clothed doesn't cover
    mannequin_exposed = np.logical_and(mannequin_visible, ~clothed_visible)
    exposed_count = mannequin_exposed.sum()

    if exposed_count > 0:
        # ANY exposed mannequin pixels = not aligned (need warping)
        return False

    # Also check IoU for general alignment
    intersection = np.logical_and(clothed_visible, mannequin_visible).sum()
    union = np.logical_or(clothed_visible, mannequin_visible).sum()

    if union == 0:
        return True

    iou = intersection / union
    return iou >= threshold
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -m pytest tests/test_optical_flow.py::test_images_already_aligned_detects_alpha_mismatch -v`

Expected: PASS

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_clothing_gen/optical_flow.py tests/test_optical_flow.py
git commit -m "feat: add alpha-based alignment detection for sprite coverage"
```

---

### Task 2: Update warp_clothing_to_pose to Use Alpha-Based Alignment

**Files:**
- Modify: `sprite_clothing_gen/optical_flow.py:181-213`

**Step 1: Write the failing test**

Add to `tests/test_optical_flow.py`:

```python
def test_warp_clothing_to_pose_warps_when_alpha_mismatch(tmp_path):
    """warp_clothing_to_pose should warp when alpha channels differ."""
    from sprite_clothing_gen.optical_flow import warp_clothing_to_pose
    from PIL import Image
    import numpy as np

    # Create base with full body
    base = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
    base_arr = np.array(base)
    base_arr[20:80, 30:70, :3] = [128, 128, 128]
    base_arr[20:80, 30:70, 3] = 255
    base_path = tmp_path / "base.png"
    Image.fromarray(base_arr).save(base_path)

    # Create clothed with smaller coverage
    clothed = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
    clothed_arr = np.array(clothed)
    clothed_arr[30:80, 30:70, :3] = [139, 69, 19]
    clothed_arr[30:80, 30:70, 3] = 255
    clothed_path = tmp_path / "clothed.png"
    Image.fromarray(clothed_arr).save(clothed_path)

    output_path = tmp_path / "output.png"

    result_path, was_skipped = warp_clothing_to_pose(
        clothed_path, base_path, output_path,
        alignment_threshold=0.98
    )

    # Should NOT skip - alpha mismatch detected
    assert was_skipped == False, "Should warp when mannequin pixels exposed"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -m pytest tests/test_optical_flow.py::test_warp_clothing_to_pose_warps_when_alpha_mismatch -v`

Expected: FAIL with `assert was_skipped == False` (currently skips because grayscale IoU passes)

**Step 3: Update warp_clothing_to_pose implementation**

In `sprite_clothing_gen/optical_flow.py`, replace the alignment check in `warp_clothing_to_pose`:

```python
def warp_clothing_to_pose(
    clothed_path: Path,
    mannequin_path: Path,
    output_path: Path,
    debug_dir: Optional[Path] = None,
    alignment_threshold: float = 0.999  # Very strict - only skip for nearly identical poses
) -> Tuple[Path, bool]:
    """Warp clothed reference to match mannequin pose.

    Main entry point for clothing transfer. If images are already
    aligned (poses match), copies the clothed image directly without
    warping to preserve maximum quality.

    Args:
        clothed_path: Path to clothed reference frame
        mannequin_path: Path to base mannequin frame (target pose)
        output_path: Where to save result
        debug_dir: Optional directory for debug outputs
        alignment_threshold: IoU threshold for skip-warp optimization

    Returns:
        Tuple of (output_path, was_skipped) where was_skipped is True
        if the image was already aligned and copied without warping
    """
    # Check if already aligned using alpha channels (more accurate)
    if images_already_aligned_alpha(clothed_path, mannequin_path, alignment_threshold):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clothed_path, output_path)
        return output_path, True  # Skipped warping

    # Load images for warping
    clothed = load_image_bgr(clothed_path)
    mannequin = load_image_bgr(mannequin_path)

    # Compute flow: how pixels move from clothed to mannequin
    flow = compute_optical_flow(clothed, mannequin)

    # Warp clothed image to match mannequin pose
    warped = warp_image(clothed, flow)

    # Create mask from mannequin (where body pixels are)
    mask = create_body_mask(mannequin)

    # Blend with clean white background
    white_bg = np.ones_like(mannequin) * 255
    result = blend_with_background(warped, white_bg, mask)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_bgr(result, output_path)

    # Save debug outputs if requested
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_image_bgr(warped, debug_dir / f"warped_{output_path.stem}.png")
        cv2.imwrite(str(debug_dir / f"mask_{output_path.stem}.png"), mask)

    return output_path, False  # Did warp
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -m pytest tests/test_optical_flow.py::test_warp_clothing_to_pose_warps_when_alpha_mismatch -v`

Expected: PASS

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_clothing_gen/optical_flow.py tests/test_optical_flow.py
git commit -m "fix: use alpha-based alignment in warp_clothing_to_pose"
```

---

### Task 3: Fix Mask Validation to Use RGBA and Remove Transparent Background

**Files:**
- Modify: `sprite_clothing_gen/orchestrator_optical.py:252-320`

**Step 1: Update _validate_masks to load RGBA and apply remove_transparent_background**

Replace `_validate_masks` method in `sprite_clothing_gen/orchestrator_optical.py`:

```python
def _validate_masks(
    self,
    base_frames: List[Path],
    warped_frames: List[Path],
    masks_dir: Path
):
    """Interactive mask validation using matplotlib GUI.

    Loads base frames as RGBA to detect transparent pixels.
    After each edit, applies remove_transparent_background to clean mask.
    """
    # Import here to avoid requiring matplotlib for headless operation
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Ensure interactive backend
    except:
        pass

    # Add parent directory to path for mask_correction_tool import
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from mask_correction_tool import MaskEditor, remove_transparent_background
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("MASK VALIDATION")
    print("=" * 70)
    print()
    print("Controls:")
    print("  Left Click: Add clothing pixels (paint red)")
    print("  Right Click: Remove clothing pixels (erase)")
    print("  Scroll: Adjust brush size")
    print("  Ctrl + Scroll: Zoom in/out")
    print("  Cmd+Z / Ctrl+Z: Undo")
    print("  Save Button: Accept and continue to next frame")
    print("  Cancel Button: Skip this frame")
    print()
    print("Note: Transparent background pixels are automatically excluded")
    print("=" * 70)

    for i, (base_path, warped_path) in enumerate(zip(base_frames, warped_frames)):
        mask_path = masks_dir / f"mask_{i:02d}.png"

        if not mask_path.exists():
            print(f"   Frame {i}: No mask found, skipping")
            continue

        print(f"\n   Reviewing frame {i+1}/{len(base_frames)}...")

        # Load base as RGBA to get alpha channel
        base_rgba = np.array(Image.open(base_path).convert('RGBA'))
        base_rgb = base_rgba[:, :, :3]

        # Load warped (clothed) image
        warped_img = np.array(Image.open(warped_path).convert('RGB'))

        # Load existing mask
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 128).astype(np.uint8)

        # Launch editor
        editor = MaskEditor(
            base_img=base_rgb,
            clothed_img=warped_img,
            mask=mask
        )
        plt.show()

        # Clean mask: remove any clothing labels on transparent pixels
        corrected_mask = remove_transparent_background(editor.mask, base_rgba)

        # Save corrected mask
        mask_output = (corrected_mask * 255).astype(np.uint8)
        Image.fromarray(mask_output).save(mask_path)

        cleaned_pixels = (editor.mask.sum() - corrected_mask.sum())
        if cleaned_pixels > 0:
            print(f"   → Saved mask {i+1} (removed {cleaned_pixels} transparent bg pixels)")
        else:
            print(f"   → Saved mask {i+1}")

    print()
    print("=" * 70)
    print("✓ Mask validation complete!")
    print("=" * 70)
```

**Step 2: Run pipeline to verify it works**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -u generate_sprite_clothing_optical.py --base examples/input/base.png --clothed examples/input/reference.png --output output/test_fixed.png 2>&1 | head -50`

Expected: Step 4 shows masks being generated, Step 5 launches editor

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add sprite_clothing_gen/orchestrator_optical.py
git commit -m "fix: mask validation removes transparent background pixels"
```

---

### Task 4: Run All Tests and Verify Full Pipeline

**Step 1: Run all optical flow tests**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -m pytest tests/test_optical_flow.py -v`

Expected: All tests PASS

**Step 2: Run full pipeline with skip-validation to verify alignment fix**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python -u generate_sprite_clothing_optical.py --base examples/input/base.png --clothed examples/input/reference.png --output output/test_alignment.png --skip-validation 2>&1 | grep -E "(Frame|warped|aligned)"`

Expected: Should see "(warped)" not "(already aligned, copied)" for frames with alpha mismatch

**Step 3: Commit final verification**

```bash
cd /Users/roberthyatt/Code/ComfyUI
git add -A
git commit -m "test: verify optical flow fixes for alignment and mask transparency"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add alpha-based alignment function | `optical_flow.py`, `test_optical_flow.py` |
| 2 | Update warp_clothing_to_pose to use alpha check | `optical_flow.py` |
| 3 | Fix mask validation to clean transparent pixels | `orchestrator_optical.py` |
| 4 | Run all tests and verify pipeline | - |
