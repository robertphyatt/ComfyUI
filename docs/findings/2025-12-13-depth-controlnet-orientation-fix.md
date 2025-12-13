# Depth ControlNet Orientation Fix - December 13, 2025

## Problem

The sprite clothing generation pipeline was producing characters facing the **wrong direction** (front-facing instead of back-facing) despite:
- Base mannequin sprites clearly showing back-facing poses
- ControlNet OpenPose correctly detecting skeleton keypoints
- Reference images showing correct back-facing armored characters

## Root Cause Analysis

### OpenPose Dimensionality Collapse

ControlNet OpenPose (and DWPose) uses 2D skeleton keypoints that **cannot distinguish front vs back facing**. The skeleton is the same whether the character faces toward or away from the camera - this is called "dimensionality collapse."

Example: A back-facing skeleton and front-facing skeleton produce identical 2D keypoint positions.

### Depth Maps from Flat Sprites

LeReS depth estimation was tested as an alternative, but **flat pixel art sprites have no real depth information**. The depth map shows the character silhouette but cannot encode front/back orientation because all pixels are essentially at the same depth plane.

### Denoise Level Impact

At higher denoise values (0.75-0.85), the model has enough freedom to "hallucinate" front-facing characters because:
1. The depth map doesn't constrain orientation
2. The model's prior heavily favors front-facing humanoids
3. Text prompts like "character" default to front-facing interpretations

## Solution

**Use lower denoise (0.5) with higher IPAdapter weight (1.2)** in img2img mode:

### Working Configuration

```python
# Key settings
denoise = 0.5              # Preserve 50% of original structure
ipadapter_weight = 1.2     # Compensate for lower denoise
weight_type = "style transfer"
preset = "PLUS (high strength)"

# Use img2img mode (VAEEncode base image, not EmptyLatentImage)
# Use Depth ControlNet for structure (even though depth is flat)
```

### Why This Works

1. **Lower denoise preserves base image orientation** - At 0.5 denoise, the original back-facing pose is strongly preserved in the latent space
2. **Higher IPAdapter weight** compensates for reduced transformation by applying stronger style transfer
3. **Img2img mode** starts from the actual base image, not random noise
4. **Depth ControlNet** still provides structural guidance even if it can't encode front/back

## Test Results

| Denoise | IPAdapter | Frame 05 Orientation | Notes |
|---------|-----------|---------------------|-------|
| 0.80 | 0.8 | Front (FAIL) | Too much freedom |
| 0.75 | 0.8 | Front (FAIL) | Still too much |
| 0.65 | 0.8 | Back (OK) | Better but inconsistent |
| 0.65 | 1.0 | Front (FAIL) | Frame 05 still problematic |
| **0.50** | **1.2** | **Back (SUCCESS)** | All 4 test frames correct |

Frame 05 was particularly challenging - it only preserved back-facing orientation at denoise 0.5.

## Tradeoffs

At denoise 0.5:
- ✅ Orientation preserved across all frames
- ✅ Clean white backgrounds
- ⚠️ Less brown armor color transfer (gray mannequin shows through)
- ⚠️ Style varies between frames

## Recommendations

1. **For consistent orientation**: Use denoise 0.5 + IPAdapter 1.2
2. **To improve style transfer**: Try IPAdapter weight 1.5 or post-process color correction
3. **Alternative approach**: Use the clothed reference as img2img starting point instead of mannequin

## Files Created During Testing

- `test_depth_baseline.py` - Depth-only test (txt2img)
- `test_depth_img2img.py` - Depth + img2img test
- `test_denoise_sweep.py` - Denoise sweep without IPAdapter
- `test_ipadapter_depth_img2img.py` - IPAdapter + Depth + img2img
- `test_ipadapter_denoise_sweep.py` - IPAdapter denoise sweep (0.70, 0.75, 0.80)
- `test_multiframe.py` - Multi-frame consistency test
- `test_lower_denoise_higher_ip.py` - Denoise 0.65 + IPAdapter 1.0
- `test_very_low_denoise.py` - **Winning config**: Denoise 0.5 + IPAdapter 1.2

## Key Insight

**The model's prior for human figures is so strong that even with ControlNet guidance, it will generate front-facing characters unless the base image structure is strongly preserved through low denoise values.**
