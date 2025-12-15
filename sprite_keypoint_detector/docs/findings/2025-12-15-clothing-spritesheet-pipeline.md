# Clothing Spritesheet Pipeline

**Date:** 2025-12-15

## Problem

Generate a clothing spritesheet that perfectly matches a base mannequin spritesheet frame-for-frame. Given a base animation (25 frames) and a clothed reference (25 frames in different poses), we need to:

1. Find the best matching clothed frame for each base frame
2. Transform the clothing to match the base pose exactly
3. Output a clothing-only spritesheet that layers cleanly over the base

## Solution

Built a modular Python pipeline with these stages:

### Stage 1: Frame Matching

For each base frame, find the best clothed reference:

1. **Top 5 by joint distance** - Sum of euclidean distances between 18 skeleton keypoints
2. **Score after transform** - Run alignment + rotation on each candidate, count:
   - Blue pixels (uncovered base) - primary metric to minimize
   - Red pixels (floating armor) - tiebreaker
3. **Select best** - Lowest blue, with red as tiebreaker within 5%

### Stage 2: Transform Pipeline

Four-step transform for each matched frame:

1. **Scale & Align** - Scale by 1.057x, align using mean of neck + hip offsets (better torso positioning than neck-only)

2. **Rigid Rotation** - Rotate limb segments at joints to match base skeleton angles. Uses forward kinematics on 4 limb chains (L/R arms, L/R legs).

3. **Soft-Edge Inpaint** - Fill gaps with texture borrowed from original via TPS mapping. Remove armor edge pixels before inpainting to avoid hard seams.

4. **Pixelize** - Downscale/upscale by factor 3 for consistent pixel art look.

### Stage 3: Output Assembly

- Individual frames saved to `output/frames/`
- Clothing spritesheet assembled matching base layout
- Debug overlay (clothing on base) for visual verification

## Key Findings

### Alignment Strategy

Neck-only alignment left hips misaligned. Using **mean of neck + hip offsets** provides better torso positioning while still keeping the neck (rotation pivot) reasonably aligned.

### Rotation Creates Acceptable Gaps

Rotating limb segments creates gaps at joints. This is acceptable because:
- Original problem areas (inner arm/leg edges) get covered
- New gaps are concentrated at rotation points
- Concentrated gaps are easier for inpainting to fill

Test results:
- Before inpaint: blue=1303 (after rotation)
- After inpaint: **blue=0** (all gaps filled)

### Pixelization Factor

Tested factors 1-5:
- Factor 1 (none): blue=0, but soft edges
- Factor 2: blue=108-160, slight blocking
- **Factor 3: blue=131-166, good pixel art look** (selected)
- Factor 4: blue=144-180, too blocky
- Factor 5: blue=258-268, severe artifacts

### Blue Threshold for Review

Set threshold at 2000 blue pixels post-rotation to flag frames needing manual review. Our working examples had 1200-1600 blue, which inpainted cleanly.

## Modules Created

| Module | Purpose |
|--------|---------|
| `annotations.py` | JSON schema with `{x, y, source, confidence}` per keypoint |
| `validation.py` | Geometric sanity + confidence checks |
| `matching.py` | Joint distance + pixel overlap scoring |
| `transform.py` | Scale, align, rotate, inpaint, pixelize |
| `spritesheet.py` | Auto-detect layout, split/assemble frames |
| `pipeline.py` | Main CLI orchestrator |
| `annotation_utils.py` | Utilities: retrain, reannotate, manual edit |
| `annotator.py` | Enhanced with ghost overlay for auto predictions |

## Annotation System

Extended annotation format tracks source and confidence:

```json
{
  "frame.png": {
    "keypoints": {
      "head": {"x": 249, "y": 172, "source": "manual", "confidence": 1.0},
      "neck": {"x": 252, "y": 217, "source": "auto", "confidence": 0.92}
    }
  }
}
```

Three annotation utilities:
1. **retrain** - Train model on manually confirmed annotations
2. **reannotate** - Run model on all frames, preserve manual keypoints
3. **edit** - Manual annotation with ghost predictions shown

## Usage

```bash
python -m sprite_keypoint_detector.pipeline \
  --base base_spritesheet.png \
  --reference clothed_spritesheet.png \
  --annotations annotations.json \
  --masks masks_dir/ \
  --output output/

# Outputs:
# - output/clothing.png (final spritesheet)
# - output/debug_overlay.png (clothing on base)
# - output/frames/clothing_XX.png (individual frames)
```

## What Didn't Work

### Nova Pixels XL for Pixelization

Tried using SDXL Nova Pixels model for "intelligent" pixelization. Even at denoise 0.10, it regenerated the character completely rather than preserving it. Simple downscale/upscale works better for preservation.

### Segment Shifting

Tried shifting limb segments to cover gaps before rotation. Created more gaps than it filled because uncovered pixels span entire limb inner edges, not concentrated at joints.

### Neck-Only Alignment

Left hips misaligned, causing torso mismatch. Mean of neck + hip works better.

## Test Results

6 integration tests passing:
- Transform produces output
- Spritesheet split/assemble roundtrip
- Joint distance calculations
- Validation checks (bounds, confidence)
