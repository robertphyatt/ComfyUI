# Early Palette Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantize frames to a shared 16-color palette early in the pipeline to simplify edge detection during inpainting, then apply final palette cleanup after pixelization.

**Architecture:** Extract a global 16-color palette from all clothed frames using k-means, quantize frames immediately after alignment, detect outline pixels by context (dark color + adjacent to transparency/bright pixels) during inpainting, and move final palette remap to after pixelization to clean up interpolation artifacts.

**Tech Stack:** Python, numpy, scikit-learn (KMeans), OpenCV, scipy

---

### Task 1: Add early palette extraction to pipeline.py

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Update imports and add early palette extraction**

Find the section after frame loading (around line 320-330) where frames are being processed. Add early palette extraction BEFORE the transform loop.

Find this block (approximately):
```python
        # Process each frame
        inpainted_frames = []
        frame_indices = []
```

Add BEFORE it:
```python
        # === Early Palette Quantization ===
        # Extract global palette from all clothed frames for consistent colors
        print("\n=== Extracting Global Palette ===")

        # Collect all clothed frames for palette extraction
        clothed_frames_for_palette = []
        for annotation in frame_annotations:
            clothed_path = frames_dir / annotation["clothed"]["filename"]
            clothed_img = cv2.imread(str(clothed_path), cv2.IMREAD_UNCHANGED)
            if clothed_img is not None:
                clothed_frames_for_palette.append(clothed_img)

        global_palette = extract_palette(clothed_frames_for_palette, n_colors=16)
        print(f"  Extracted {len(global_palette)}-color global palette")

        # Save palette visualization early
        if debug:
            save_palette_image(global_palette, debug_dir / "palette.png")
            print(f"  Saved palette visualization to debug/palette.png")
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import run_pipeline; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): extract global palette early before transforms

Extract 16-color palette from all clothed frames before processing.
This enables simpler edge detection during inpainting.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add quantize function to color_correction.py

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/color_correction.py`

**Step 1: Add quantize_frame function**

Add this function after `remap_frame_to_palette`:

```python
def quantize_frame(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Quantize a frame to use only palette colors.

    This is an alias for remap_frame_to_palette, used for clarity
    when quantizing early in the pipeline vs final cleanup.

    Args:
        frame: BGRA image
        palette: Array of shape (n_colors, 3) with BGR values

    Returns:
        Quantized BGRA image using only palette colors
    """
    return remap_frame_to_palette(frame, palette)
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.color_correction import quantize_frame; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add color_correction.py
git commit -m "feat(color_correction): add quantize_frame function

Alias for remap_frame_to_palette for semantic clarity when
quantizing early vs final palette cleanup.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Quantize clothed frames after alignment in pipeline

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Update imports**

Find:
```python
from .color_correction import extract_palette, remap_all_frames, save_palette_image
```

Replace with:
```python
from .color_correction import extract_palette, remap_frame_to_palette, save_palette_image, quantize_frame
```

**Step 2: Add quantization after alignment in transform loop**

Inside the frame processing loop, find where `aligned_clothed` is created (in transform_frame_debug or after scale_and_align). Add quantization right after alignment.

Find this pattern in the loop (around line 380-400):
```python
            debug_output = transform_frame_debug(
                clothed_img, clothed_kpts_array,
                base_img, base_kpts_array,
                mask_img, config
            )
```

This is inside `transform_frame_debug`. We need to modify the transform functions to accept an optional palette parameter.

Actually, better approach: quantize the clothed image BEFORE passing to transform_frame_debug.

Find where clothed_img is loaded in the loop:
```python
            clothed_img = cv2.imread(str(clothed_path), cv2.IMREAD_UNCHANGED)
```

Add RIGHT AFTER:
```python
            # Quantize clothed image to global palette
            clothed_img = quantize_frame(clothed_img, global_palette)
```

**Step 3: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import run_pipeline; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): quantize clothed frames to global palette before transform

Each clothed frame is now quantized to the shared 16-color palette
immediately after loading, before any transforms are applied.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Add context-based outline detection to transform.py

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`

**Step 1: Add function to detect outline pixels**

Add this function before `apply_inpaint`:

```python
def detect_outline_pixels(
    image: np.ndarray,
    brightness_threshold: int = 60,
    contrast_threshold: int = 50
) -> np.ndarray:
    """Detect outline pixels that should be inpainted over.

    An outline pixel is one that:
    1. Is relatively dark (below brightness_threshold), AND
    2. Is adjacent to transparency OR adjacent to a much brighter pixel

    This preserves dark interior details (shadows, leather) while
    catching outline artifacts from rotation.

    Args:
        image: BGRA image (quantized to palette)
        brightness_threshold: Max brightness to consider as potential outline (0-255)
        contrast_threshold: Min brightness difference to neighbor to trigger

    Returns:
        Boolean mask where True = outline pixel to inpaint
    """
    h, w = image.shape[:2]
    alpha = image[:, :, 3]
    visible = alpha > 128

    # Calculate brightness (simple average of BGR)
    brightness = np.mean(image[:, :, :3], axis=2).astype(np.float32)

    # Dark pixels are candidates
    is_dark = brightness < brightness_threshold

    # Check adjacency to transparency
    transparent = alpha <= 128
    adjacent_to_transparent = binary_dilation(transparent, iterations=1) & visible

    # Check adjacency to much brighter pixels
    # Dilate brightness and check if any neighbor is much brighter
    from scipy.ndimage import maximum_filter
    max_neighbor_brightness = maximum_filter(brightness, size=3)
    adjacent_to_bright = (max_neighbor_brightness - brightness) > contrast_threshold

    # Outline = dark AND (adjacent to transparency OR adjacent to bright)
    outline_mask = is_dark & visible & (adjacent_to_transparent | adjacent_to_bright)

    return outline_mask
```

**Step 2: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import detect_outline_pixels; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): add context-based outline pixel detection

Detects outline pixels by finding dark pixels adjacent to transparency
or much brighter neighbors. These pixels will be inpainted over to
remove 'double arm' artifacts from rotation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Integrate outline detection into apply_inpaint

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`

**Step 1: Modify apply_inpaint to include outline pixels in inpaint region**

Find in `apply_inpaint` (around line 369):
```python
    inpaint_region = uncovered | armor_edge
```

Replace with:
```python
    # Detect outline pixels that should be inpainted over
    # These are dark pixels adjacent to transparency or bright pixels
    outline_pixels = detect_outline_pixels(armor)

    inpaint_region = uncovered | armor_edge | outline_pixels
```

**Step 2: Clear outline pixels from result before inpainting**

Find (around line 367):
```python
    result = armor.copy()
    result[:, :, 3] = np.where(armor_edge, 0, armor[:, :, 3])
```

Replace with:
```python
    result = armor.copy()
    # Clear both armor edge and outline pixels - they'll be inpainted
    clear_mask = armor_edge | outline_pixels
    result[:, :, 3] = np.where(clear_mask, 0, armor[:, :, 3])
```

Wait - we need to detect outline_pixels BEFORE using it. Let me fix the order:

Find the section starting with:
```python
    armor_edge = _get_armor_edge_near_uncovered(armor[:, :, 3], uncovered, config.edge_width)

    result = armor.copy()
    result[:, :, 3] = np.where(armor_edge, 0, armor[:, :, 3])

    inpaint_region = uncovered | armor_edge
```

Replace the ENTIRE section with:
```python
    armor_edge = _get_armor_edge_near_uncovered(armor[:, :, 3], uncovered, config.edge_width)

    # Detect outline pixels that should be inpainted over
    # These are dark pixels adjacent to transparency or bright pixels
    outline_pixels = detect_outline_pixels(armor)

    result = armor.copy()
    # Clear both armor edge and outline pixels - they'll be inpainted
    clear_mask = armor_edge | outline_pixels
    result[:, :, 3] = np.where(clear_mask, 0, armor[:, :, 3])

    inpaint_region = uncovered | armor_edge | outline_pixels
```

**Step 3: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import apply_inpaint; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): inpaint over detected outline pixels

Outline pixels (dark + adjacent to transparency/bright) are now
included in the inpaint region, removing 'double arm' artifacts.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Move final palette remap to after pixelization

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/pipeline.py`

**Step 1: Find and remove current palette remapping location**

Find the current palette remapping section (around line 430-450):
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

Remove this entire block (we extracted palette earlier and will remap after pixelization).

**Step 2: Update pixelization to use inpainted_frames directly**

Find:
```python
        # === Pixelization ===
        print("\n=== Applying Pixelization ===")
```

The pixelization should operate on `inpainted_frames` now (not `corrected_frames`).

Find the pixelization loop and ensure it uses `inpainted_frames`:
```python
        final_frames = []
        for frame in inpainted_frames:
            pixelized = apply_pixelize(frame, config.pixelize_factor)
            final_frames.append(pixelized)
```

**Step 3: Add final palette remap AFTER pixelization**

After the pixelization loop, add:
```python
        # === Final Palette Cleanup ===
        # Remap to palette after pixelization to clean up any interpolation artifacts
        print("\n=== Final Palette Cleanup ===")
        final_frames = [remap_frame_to_palette(f, global_palette) for f in final_frames]
        print(f"  Remapped {len(final_frames)} frames to global palette")

        # Save final frames to debug if enabled
        if debug:
            for i, (final, base_idx) in enumerate(zip(final_frames, frame_indices)):
                cv2.imwrite(str(debug_dir / "6_final" / f"frame_{base_idx:02d}.png"), final)
            print(f"  Saved final frames to debug/6_final/")
```

**Step 4: Remove the intermediate 5_palette_remapped debug directory creation**

Find:
```python
            (debug_dir / "5_palette_remapped").mkdir(exist_ok=True)
```

Remove this line (we no longer have this intermediate step).

**Step 5: Update comparison image step**

Find in the comparison steps list:
```python
        ("5_palette_remapped", "Palette"),
```

Remove this line - we no longer have this intermediate step. The final frames will show the palette-cleaned result.

**Step 6: Verify syntax**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.pipeline import run_pipeline; print('OK')"`

Expected: `OK`

**Step 7: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "feat(pipeline): move final palette remap to after pixelization

Pipeline order is now:
1. Extract global palette from clothed frames
2. Quantize clothed frames to palette
3. Transform (align, mask, rotate, inpaint)
4. Pixelize
5. Final palette cleanup (removes interpolation artifacts)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Revert erosion change from earlier

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/transform.py`

**Step 1: Revert erosion back to 2**

Find:
```python
    # Create interior mask for safe sampling (avoid edge pixels)
    # Use armor's actual alpha, not the mask - we want to avoid armor's edge pixels
    # Erosion=5 to avoid sampling from outline pixels that cause "double arm" artifacts
    interior_mask = get_interior_mask(armor[:, :, 3], erosion=5)
```

Replace with:
```python
    # Create interior mask for safe sampling (avoid edge pixels)
    # Use armor's actual alpha, not the mask - we want to avoid armor's edge pixels
    interior_mask = get_interior_mask(armor[:, :, 3], erosion=2)
```

**Step 2: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "fix(transform): revert erosion to 2, outline detection handles edges

The new outline pixel detection handles the double-arm artifact,
so aggressive erosion is no longer needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Test full pipeline

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
1. Pipeline extracts global palette early
2. Clothed frames are quantized before transform
3. Outline pixels are detected and inpainted over
4. Pixelization runs
5. Final palette cleanup runs
6. Output saved with debug directories

**Step 2: Verify debug output**

```bash
ls /tmp/pipeline_test/debug/
```

Expected: `1_aligned/`, `2_masked/`, `3_rotated/`, `4_inpainted/`, `6_final/`, `comparison/`, `palette.png`

(Note: no `5_palette_remapped/` - that step is now combined with final)

**Step 3: Check frames 00 and 01 for double-arm artifact**

```bash
open /tmp/pipeline_test/debug/comparison/frame_00.png
open /tmp/pipeline_test/debug/comparison/frame_01.png
```

Expected: No hard edge line on the right arm - the outline artifact should be gone.

**Step 4: Verify palette consistency**

```bash
open /tmp/pipeline_test/debug/palette.png
```

Expected: 4x4 grid showing the 16 extracted colors used throughout the pipeline.

---

## Summary

This plan implements:
1. **Early palette extraction** - Global 16-color palette from all clothed frames
2. **Early quantization** - Clothed frames quantized before transforms
3. **Context-based outline detection** - Dark pixels adjacent to transparency/bright pixels
4. **Outline inpainting** - Detected outlines are inpainted over
5. **Final palette cleanup** - After pixelization to remove interpolation artifacts
6. **Revert erosion** - No longer needed with outline detection

The pipeline order becomes:
1. Load frames
2. Extract global palette
3. Quantize clothed frames
4. Transform (align → mask → rotate → inpaint with outline detection)
5. Pixelize
6. Final palette remap
7. Output
