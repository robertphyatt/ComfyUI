# IPAdapter + ControlNet Clothing Generation Design

**Date:** 2025-12-09
**Status:** Approved for implementation

## Problem Statement

Previous approach (skeleton-guided warping) destroyed pixel art quality, creating blur, ghosting, and distortion. Need a way to generate clothing in the exact base character poses without post-generation alignment.

## Solution Overview

Use ComfyUI's IPAdapter + OpenPose ControlNet to generate clothed characters directly in base poses:
- **IPAdapter** learns armor appearance from all 25 clothed reference frames
- **OpenPose ControlNet** enforces exact base skeleton pose
- **Inpainting** generates only in character region
- **Existing masking system** extracts clothing layer

## Architecture

### Input Preparation (Per Frame)

**Required inputs:**
1. Base character image (512x512 RGBA) - gray body sprite
2. Auto-generated inpainting mask - `mask = (alpha > 0) * 255`
3. Base skeleton - OpenPose keypoints from base frame

**One-time setup:**
4. IPAdapter reference images - all 25 clothed frames
5. Text prompt - "Brown leather armor with shoulder pauldrons, chest plate, arm guards, leg armor, fantasy RPG character, pixel art style, detailed"
6. Negative prompt - "blurry, low quality, distorted, multiple heads, deformed, extra limbs"

### ComfyUI Workflow

```
[Load Base Image] → [Extract Alpha Mask] → [Inpaint Mask]
                                              ↓
[Load 25 Clothed Refs] → [IPAdapter Encode] → [IPAdapter Apply]
                                                ↓
[Base Image] → [OpenPose Preprocessor] → [ControlNet (OpenPose)]
                                           ↓
[Text Prompt] → [CLIP Encode] --------→ [KSampler (Inpainting)]
[Negative Prompt] → [CLIP Encode] -----→      ↓
                                          [VAE Decode]
                                               ↓
                                          [Clothed Character Output]
```

**Key workflow nodes:**
- **IPAdapter**: Batch loads 25 clothed frames, learns armor style/appearance
- **OpenPose ControlNet**: Extracts skeleton from base, enforces pose
- **KSampler (Inpainting)**: Generates only in masked region, preserves background
- **VAE Decode**: Converts latent to pixel image

### Generation Parameters

**KSampler settings:**
- Steps: 30-40 (sufficient for detail convergence)
- CFG Scale: 7.0 (strong prompt adherence)
- Sampler: DPM++ 2M Karras (good for pixel art)
- Denoise: 1.0 (full generation in masked area)
- Seed: Fixed per frame (reproducible results)

### Post-Processing Pipeline

**After generation (per frame):**

1. **Save generated image:**
   - Output: `training_data/frames_ipadapter_generated/clothed_frame_XX.png`

2. **Run through existing masking:**
   ```bash
   python predict_masks_with_model.py \
     --frames-dir training_data/frames_ipadapter_generated \
     --output-dir training_data/masks_ipadapter
   ```

3. **Extract clothing layers:**
   ```bash
   python extract_clothing_final.py \
     --frames-dir training_data/frames_ipadapter_generated \
     --masks-dir training_data/masks_ipadapter \
     --output clothing_spritesheet_ipadapter.png
   ```

4. **Create final spritesheet:**
   - Combine 25 clothing layers into 5x5 grid
   - Output: Transparent PNG with only clothing pixels

## Key Design Decisions

### Decision 1: IPAdapter vs LoRA Training
**Chosen:** IPAdapter (Option B from brainstorming)
**Reason:** No training required, works with existing models, 25 images sufficient as references

### Decision 2: Full Body vs Precise Masking
**Chosen:** Full body mask (Option B), with fallback to precise masks (Option A)
**Reason:** Simpler implementation, let model decide coverage. If it generates armor on head/hands, switch to body-part-specific masks.

### Decision 3: Extraction Method
**Chosen:** Use existing trained U-Net masking system
**Reason:** Already proven to work, no need to reinvent

### Decision 4: Inpainting vs Full Generation
**Chosen:** Inpainting approach (Option C from workflow brainstorming)
**Reason:** User insight - inverted character masks give exactly where clothing should go, more precise than full generation

## Benefits

1. **No warping artifacts** - Generated in correct pose from start
2. **Preserves pixel art quality** - No interpolation/blending
3. **Proven masking** - Reuses existing U-Net system
4. **Flexible** - IPAdapter learns from all 25 angles
5. **Reproducible** - Fixed seeds ensure consistent results

## Fallback Plans

**If full body mask fails (generates armor on head/hands):**
- Switch to precise body-part masks
- Mask only: chest, shoulders, arms, legs
- Exclude: head, hands, feet

**If IPAdapter averaging loses details:**
- Switch to Option C (generate 25 times with closest reference each)
- Match each base pose to nearest clothed pose
- Use single reference per generation

## Success Criteria

1. Generated clothing matches base skeleton pose exactly
2. Pixel art quality preserved (crisp, no blur)
3. Armor appearance matches reference frames
4. No body parts peeking through clothing
5. Clean extraction using existing masking system

## Implementation Notes

- ComfyUI server must be running
- Requires IPAdapter model loaded
- Requires OpenPose ControlNet model
- Generated frames go through same pipeline as original workflow
- Final spritesheet format identical to previous approach
