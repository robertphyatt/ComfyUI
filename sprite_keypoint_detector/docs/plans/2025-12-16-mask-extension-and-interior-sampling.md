# Mask Extension and Interior Sampling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix edge artifacts by (1) extending masks to include thin edge strips near transparency, and (2) only sampling from interior pixels during inpainting.

**Architecture:** Add `extend_mask_to_edges()` function before mask application, and `get_interior_mask()` function for inpainting. Both use morphological operations (dilation/erosion) from scipy.ndimage.

**Tech Stack:** Python, numpy, scipy.ndimage (binary_dilation, binary_erosion)

---

### Task 1: Add mask extension function

**Files:**
- Modify: `sprite_keypoint_detector/transform.py`

**Step 1: Add the extend_mask_to_edges function after apply_mask**

Add this function after line 120 (after `apply_mask`):

```python
def extend_mask_to_edges(
    mask: np.ndarray,
    clothed_alpha: np.ndarray,
    max_gap: int = 3
) -> np.ndarray:
    """Extend mask to fill thin gaps between mask edge and transparent pixels.

    When the armor mask is slightly smaller than the clothed sprite silhouette,
    thin strips of edge pixels get excluded. This extends the mask to include
    those pixels, but ONLY if they're near transparency (not mannequin pixels
    in the middle of the sprite).

    Args:
        mask: Original armor mask (0 or 255)
        clothed_alpha: Alpha channel of clothed image
        max_gap: Maximum gap width to fill (pixels)

    Returns:
        Extended mask
    """
    from scipy.ndimage import binary_dilation

    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask_visible = mask > 128
    clothed_visible = clothed_alpha > 128

    # Gap candidates: pixels in clothed sprite but not in mask
    gap_candidates = clothed_visible & ~mask_visible

    # Only fill gaps that are near transparency (edge of sprite)
    # This prevents extending into mannequin pixels in the middle
    transparent = clothed_alpha < 128
    near_transparency = binary_dilation(transparent, iterations=max_gap)

    # Thin strip = gap candidates that are near transparency
    thin_strip = gap_candidates & near_transparency

    # Extended mask = original + thin strips
    extended = mask_visible | thin_strip

    return (extended * 255).astype(np.uint8)
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import extend_mask_to_edges; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): add extend_mask_to_edges function

Extends armor mask to include thin edge strips (up to 3px) between
the mask edge and transparent pixels. Only extends near transparency,
not into mannequin pixels in the middle of the sprite.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Integrate mask extension into transform pipeline

**Files:**
- Modify: `sprite_keypoint_detector/transform.py`

**Step 1: Update transform_frame to use mask extension**

Find lines ~490-494 in transform_frame (after scaling the mask):

```python
    aligned_mask, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config)
    scaled_mask = aligned_mask[:, :, 0]

    # Extract armor
    armor = apply_mask(aligned_clothed, scaled_mask)
```

Replace with:

```python
    aligned_mask, _ = scale_and_align(mask_rgba, clothed_kpts, base_kpts, config)
    scaled_mask = aligned_mask[:, :, 0]

    # Extend mask to include thin edge strips near transparency
    extended_mask = extend_mask_to_edges(scaled_mask, aligned_clothed[:, :, 3])

    # Extract armor
    armor = apply_mask(aligned_clothed, extended_mask)
```

**Step 2: Update transform_frame_debug the same way**

Find the same pattern in transform_frame_debug (~lines 544-548) and apply the same change.

**Step 3: Test the pipeline**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --frames-dir training_data/frames/ --annotations training_data/annotations.json --masks training_data/masks_corrected/ --output /tmp/pipeline_test/ --skip-validation --debug 2>&1 | tail -5`

Expected: Pipeline completes successfully

**Step 4: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): integrate mask extension into pipeline

Apply extend_mask_to_edges() after scaling the mask to include thin
edge strips before extracting armor. This prevents edge pixels from
being excluded and then badly inpainted.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Add interior mask function for inpainting

**Files:**
- Modify: `sprite_keypoint_detector/transform.py`

**Step 1: Add get_interior_mask function before apply_inpaint**

Add this function around line 262 (before apply_inpaint):

```python
def get_interior_mask(alpha: np.ndarray, erosion: int = 2) -> np.ndarray:
    """Get mask of interior pixels safe to sample from (not near edges).

    Args:
        alpha: Alpha channel (0-255)
        erosion: How many pixels to erode from edges

    Returns:
        Boolean mask where True = interior pixel
    """
    from scipy.ndimage import binary_erosion

    visible = alpha > 128
    interior = binary_erosion(visible, iterations=erosion)

    return interior
```

**Step 2: Verify it compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.transform import get_interior_mask; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "feat(transform): add get_interior_mask function

Creates a mask of interior pixels by eroding the alpha channel.
Used by inpainting to avoid sampling from edge pixels.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update inpainting to use interior sampling

**Files:**
- Modify: `sprite_keypoint_detector/transform.py`

**Step 1: Modify apply_inpaint to create interior mask**

In apply_inpaint, after line ~288 (`neck_y = int(base_kpts[1, 1])`), add:

```python
    # Create interior mask for safe sampling (avoid edge pixels)
    interior_mask = get_interior_mask(armor_mask, erosion=2)
```

**Step 2: Update TPS sampling check**

Change line ~323 from:

```python
            if armor_mask[src_y, src_x] > 128:
```

To:

```python
            if armor_mask[src_y, src_x] > 128 and interior_mask[src_y, src_x]:
```

**Step 3: Update fallback sampling to prefer interior pixels**

Change the fallback loop (~lines 329-341). Replace:

```python
        # Fallback: nearest armor pixel
        orig_armor_alpha = armor[:, :, 3] > 128
        for radius in range(1, 30):
            y1, y2 = max(0, dst_y - radius), min(h, dst_y + radius + 1)
            x1, x2 = max(0, dst_x - radius), min(w, dst_x + radius + 1)
            box = orig_armor_alpha[y1:y2, x1:x2]
            if np.any(box):
                box_ys, box_xs = np.where(box)
                abs_ys, abs_xs = box_ys + y1, box_xs + x1
                distances = (abs_ys - dst_y) ** 2 + (abs_xs - dst_x) ** 2
                closest = np.argmin(distances)
                result[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                result[dst_y, dst_x, 3] = 255
                break
```

With:

```python
        # Fallback: nearest interior armor pixel (prefer interior, fall back to any)
        orig_armor_alpha = armor[:, :, 3] > 128
        for radius in range(1, 30):
            y1, y2 = max(0, dst_y - radius), min(h, dst_y + radius + 1)
            x1, x2 = max(0, dst_x - radius), min(w, dst_x + radius + 1)

            # Prefer interior pixels
            box_interior = orig_armor_alpha[y1:y2, x1:x2] & interior_mask[y1:y2, x1:x2]
            if np.any(box_interior):
                box_ys, box_xs = np.where(box_interior)
                abs_ys, abs_xs = box_ys + y1, box_xs + x1
                distances = (abs_ys - dst_y) ** 2 + (abs_xs - dst_x) ** 2
                closest = np.argmin(distances)
                result[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                result[dst_y, dst_x, 3] = 255
                break

            # Fall back to any armor pixel if no interior found
            box = orig_armor_alpha[y1:y2, x1:x2]
            if np.any(box):
                box_ys, box_xs = np.where(box)
                abs_ys, abs_xs = box_ys + y1, box_xs + x1
                distances = (abs_ys - dst_y) ** 2 + (abs_xs - dst_x) ** 2
                closest = np.argmin(distances)
                result[dst_y, dst_x, :3] = armor[abs_ys[closest], abs_xs[closest], :3]
                result[dst_y, dst_x, 3] = 255
                break
```

**Step 4: Test the pipeline**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --frames-dir training_data/frames/ --annotations training_data/annotations.json --masks training_data/masks_corrected/ --output /tmp/pipeline_test/ --skip-validation --debug 2>&1 | tail -5`

Expected: Pipeline completes successfully

**Step 5: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add transform.py
git commit -m "fix(inpaint): only sample from interior pixels

When inpainting gaps, avoid sampling from edge pixels which can create
artifacts like duplicate arm outlines. TPS sampling now requires the
source pixel to be in the interior mask. Fallback sampling prefers
interior pixels but falls back to any armor pixel if needed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Verify fixes visually

**Files:**
- None (verification only)

**Step 1: Check frame 00 comparison**

Open: `/tmp/pipeline_test/debug/comparison/frame_00.png`

Expected: Right arm/hand should no longer have dark edge artifact running through middle

**Step 2: Check frame 01 comparison**

Open: `/tmp/pipeline_test/debug/comparison/frame_01.png`

Expected: Right arm/hand should be clean without duplicate edge artifacts

**Step 3: Push all changes**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git push
```

---

## Summary

Two complementary fixes:

1. **Mask extension** (Tasks 1-2): Extends the armor mask by up to 3 pixels to include thin edge strips near transparency. This prevents edge pixels from being excluded and needing inpainting.

2. **Interior sampling** (Tasks 3-4): When inpainting is needed, only sample from interior pixels (2px erosion from edges). This prevents edge textures from being painted into the middle of body parts.

The mask extension is the primary fix (addresses root cause). Interior sampling is a safety net for cases where inpainting still happens.
