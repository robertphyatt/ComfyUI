# ControlNet Tile vs OpenPose Comparison

**Date:** 2025-12-09
**Test:** ControlNet Tile vs OpenPose for pixel art sprite generation

## Test Configuration

**Setup:**
- IPAdapter + ControlNet (Tile vs OpenPose)
- Prompt: "character wearing brown leather armor, pixel art"
- Seed: 777 (Tile), baseline (OpenPose)
- Model: v1-5-pruned-emaonly.safetensors
- Steps: 20, CFG: 7.0, denoise: 1.0 (txt2img)

**Reference Images:**
- Clothed: `input/clothed_frame_00.png`
- Base: `training_data/frames/base_frame_00.png`

## Results

### ControlNet Tile
**Output:** `output/test_controlnet_tile_00001_.png`

**Visual Assessment:**
- Generated BLUE/GRAY ARMOR instead of brown leather
- Does NOT match the prompt specification
- Background is noisy with horizontal stripes/artifacts
- Pixel edges are reasonably clean
- Character pose appears correct
- Armor style is metallic/plate-like rather than leather

**Silhouette Accuracy:**
- Character shape is preserved
- Proportions appear correct

**Edge Quality:**
- Pixel edges are relatively clean
- Some noise in the background

### ControlNet OpenPose
**Output:** `output/ipadapter_test_with_controlnet_00001_.png`

**Visual Assessment:**
- Generated BROWN LEATHER ARMOR as specified in prompt
- CORRECTLY matches the prompt specification
- Background is clean, solid brown color
- Pixel edges are sharp and clean
- Character pose appears correct
- Armor style matches "leather armor" aesthetic

**Silhouette Accuracy:**
- Character shape is preserved
- Proportions appear correct

**Edge Quality:**
- Pixel edges are sharp and clean
- No background noise
- Overall cleaner output

## Analysis

**Prompt Adherence:**
- OpenPose: Correctly generated brown leather armor
- Tile: Generated wrong color/material (blue/gray metal instead of brown leather)

**Background Quality:**
- OpenPose: Clean, solid background
- Tile: Noisy background with horizontal stripe artifacts

**Pose Accuracy:**
- Both variants preserve the character pose correctly

**Gemini's Research Claim:**
> "ControlNet Tile is superior for pixel art sprites because OpenPose fails on few-pixel-wide limbs"

**Reality Check:**
- OpenPose does NOT fail on this pixel art character
- OpenPose correctly interprets the pose
- Tile produces INCORRECT colors and noisy background
- No evidence of Tile being superior for this use case

## Recommendation

**Winner: ControlNet OpenPose**

**Rationale:**
1. OpenPose correctly follows the prompt (brown leather armor)
2. OpenPose produces cleaner background
3. OpenPose maintains sharp pixel edges
4. Tile fails to generate correct armor color/material
5. No observed benefit from Tile for this pixel art style

**Decision:**
- Keep using ControlNet OpenPose in production workflow
- Do NOT switch to ControlNet Tile
- Gemini's research suggestion does not apply to this specific pixel art style

## Next Steps

- Proceed with Task 2: Test img2img with denoise=0.65
- Continue using OpenPose variant in production
