# Semantic Segmentation + Edge Refinement Design

**Date:** 2025-12-07
**Problem:** Bounding box approach removes too much. AI guidance cannot prevent shoulder/armor removal.
**Solution:** Pixel-level semantic segmentation with edge-based boundary refinement.

## Problem Analysis

The hybrid bounding box approach fails because:

1. **AI guidance is imprecise** - "Only the HEAD" still produces boxes that include shoulders
2. **Tolerance matching hits highlights** - Armor highlights (RGB 190) match skin (RGB 180) within tolerance
3. **Semantic vs geometric mismatch** - AI understands semantics poorly; bounding boxes capture geometry poorly

We asked AI to draw boxes around heads. AI drew boxes around heads plus shoulders. No amount of prompting fixes this reliably.

## Core Insight

Separate semantic understanding from geometric precision:

- **AI:** Classify regions semantically (clothing vs base character)
- **Edges:** Define boundaries geometrically (where colors change)
- **Combination:** Semantic regions snapped to geometric boundaries

## Pipeline Architecture

### Step 1: Downscale for AI Processing

```python
clothed_256 = clothed_frame.resize((256, 256), Image.LANCZOS)
```

Reduces AI workload by 4x. Downscaling preserves semantic content while reducing detail.

### Step 2: AI Segmentation Mask

**Prompt:**
```
Classify each pixel as CLOTHING (1) or BASE (0).
Output run-length encoded mask:
{
  "mask": [
    {"value": 0, "count": 1234},
    {"value": 1, "count": 567}
  ]
}
```

**Run-length encoding:** Compresses 65,536 pixels into ~hundreds of runs. Fits in AI response limits. Decodes to full 256×256 mask.

**Result:** Rough semantic classification at low resolution.

### Step 3: Upscale Rough Mask

```python
mask_512_rough = cv2.resize(mask_256, (512, 512), cv2.INTER_NEAREST)
```

Nearest-neighbor preserves binary values. Creates blocky but semantically correct mask.

### Step 4: Edge Detection on Difference

```python
diff = cv2.absdiff(clothed_gray, base_gray)
edges = cv2.Canny(diff, threshold1=50, threshold2=150)
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)
```

**Why difference:** Edges where clothing was added. Ignores base character anatomy.

**Dilation:** Creates 2-3 pixel search zones around edges. Forgives small AI errors.

### Step 5: Boundary Snapping

```python
mask_boundaries = find_mask_boundaries(mask_512_rough)

for y, x in mask_boundaries:
    nearest_edge = find_nearest_edge(edges_dilated, (y, x), search_radius=10)

    if nearest_edge:
        snap_boundary_to_edge(mask_512_rough, (y, x), nearest_edge)
```

**Algorithm:** For each pixel where rough mask transitions, search within 10 pixels for nearest detected edge. Snap mask boundary to that edge.

**Result:** Semantic regions aligned to precise visual boundaries.

### Step 6: Morphological Cleanup

```python
# Fill holes in clothing regions
mask_filled = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=2)

# Remove isolated base character pixels
mask_cleaned = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=1)
```

**Close operation:** Fills small holes (AI classified pixel as "base" when surrounded by "clothing")

**Open operation:** Removes small islands (AI classified pixel as "clothing" when surrounded by "base")

### Step 7: Apply Final Mask

```python
clothing_arr[:, :, 3] = np.where(mask_cleaned == 1,
                                  clothed_arr[:, :, 3],  # Keep clothing
                                  0)                     # Remove base
```

Sets alpha channel: clothing pixels preserved, base character pixels transparent.

## Why This Works

**AI handles semantics at 256×256:**
- "This region is a shoulder pad" (clothing)
- "This region is gray skin" (base character)
- Fast, reliable, low-resolution semantic decisions

**Edge detection handles geometry at 512×512:**
- "Sharp color transition at pixel (243, 156)"
- Precise, deterministic, high-resolution boundaries

**Refinement combines both:**
- AI: "Shoulder pad extends to approximately here"
- Edges: "Shoulder pad edge is exactly at this pixel"
- Snap: "Mask boundary moves to exact edge location"

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| AI resolution | 256×256 | 4x reduction, manageable output, preserves semantics |
| Canny threshold1 | 50 | Weak edges, sensitive to subtle boundaries |
| Canny threshold2 | 150 | Strong edges, filters noise |
| Edge dilation | 1 iteration | Creates 2-3 pixel search zones |
| Snap radius | 10 pixels | Tolerates AI errors while preventing bad snaps |
| Close iterations | 2 | Fills holes up to ~4 pixels |
| Open iterations | 1 | Removes islands up to ~2 pixels |

## Risks and Mitigations

**Risk:** Ollama's ministral-3:8b cannot produce run-length encoded output
**Mitigation:** Test AI capability first. Fall back to coordinate list if needed.

**Risk:** Edge detection on diff misses boundaries where lighting changed
**Mitigation:** Canny is robust to lighting. Dilation creates forgiving search zones.

**Risk:** Snapping algorithm is too slow (O(boundaries × edges))
**Mitigation:** Use spatial indexing (KD-tree) for edge lookups if needed.

## Success Criteria

1. Frame 0 shoulder armor preserved (no white gaps)
2. Frame 0 gray head completely removed
3. Consistent results across all 25 frames
4. No tolerance tuning required
5. No AI guidance prompting games

## Implementation Notes

**Libraries:** OpenCV for edge detection and morphology, PIL for image handling, NumPy for array operations.

**Testing:** Prototype AI segmentation first. Validate run-length encoding works before implementing full pipeline.

**Fallback:** If AI segmentation fails, fall back to current bounding box approach with reduced tolerance.
