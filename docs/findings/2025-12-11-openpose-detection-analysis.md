# OpenPose Detection Analysis: Frame 4 Success vs Frames 0/12 Failure

**Date:** 2025-12-11
**Goal:** Identify why OpenPose successfully detects poses on frame 4 but fails on frames 0 and 12

## Test Setup

**Frames Analyzed:**
- Frame 00 (`base_frame_00.png`) - FAILING
- Frame 04 (`base_frame_04.png`) - WORKING
- Frame 12 (`base_frame_12.png`) - FAILING

**Analysis Methods:**
1. Image property comparison (size, bounding box, centering)
2. Pixel-level analysis (brightness, contrast, color distribution)
3. OpenPose skeleton output verification
4. Multi-configuration OpenPose testing (default, body-only, high-res)

## Detection Verification Results

### Skeleton Output Analysis

**Frame 00 (FAILING):**
- Non-black pixels: 0
- Coverage: 0.00%
- File size: 12,348 bytes (compressed black image)
- Result: NO DETECTION across all OpenPose configurations

**Frame 04 (WORKING):**
- Non-black pixels: 4,249
- Coverage: 1.62%
- File size: 14,186 bytes
- Result: SUCCESSFUL DETECTION across all OpenPose configurations

**Frame 12 (FAILING):**
- Non-black pixels: 0
- Coverage: 0.00%
- File size: 12,348 bytes (compressed black image)
- Result: NO DETECTION across all OpenPose configurations

### Multi-Configuration Test Results

All three frames tested with:
- Default settings (hands + body + face, resolution 512)
- Body-only mode (no hands/face detection)
- High-resolution mode (resolution 1024)

**Results:**
- Frame 00: 0/3 configurations detected (FAILING)
- Frame 04: 3/3 configurations detected (WORKING)
- Frame 12: 0/3 configurations detected (FAILING)

This confirms the issue is NOT related to OpenPose configuration parameters.

## Image Property Comparison

### Basic Properties

All three frames are identical in:
- Canvas size: 512x512 pixels
- Color mode: RGBA
- Format: PNG
- Contrast range: 0-255 (full range)

### Sprite Bounding Box

**Frame 00:**
- Bounding box: [183, 122, 343, 388]
- Sprite dimensions: 160 x 266 pixels
- Center position: (263.0, 255.0)
- Offset from canvas center: (7.0, -1.0)

**Frame 04:**
- Bounding box: [179, 121, 337, 385]
- Sprite dimensions: 158 x 264 pixels
- Center position: (258.0, 253.0)
- Offset from canvas center: (2.0, -3.0)

**Frame 12:**
- Bounding box: [183, 121, 343, 388]
- Sprite dimensions: 160 x 267 pixels
- Center position: (263.0, 254.5)
- Offset from canvas center: (7.0, -1.5)

**Differences:**
- Size difference: 2-3 pixels (negligible)
- Position difference: 5 pixels X-offset (negligible)
- These differences are TOO SMALL to explain detection failure

### Color and Brightness Analysis

**Frame 00:**
- Mean brightness: 107.28, 107.68, 119.87 (RGB)
- Standard deviation: 62.60, 62.56, 57.10
- Unique colors: 9,667
- Skin-tone pixels: 0 (0.00%)

**Frame 04:**
- Mean brightness: 108.84, 109.15, 121.45 (RGB)
- Standard deviation: 63.11, 63.17, 57.58
- Unique colors: 10,013
- Skin-tone pixels: 34 (0.14%)

**Frame 12:**
- Mean brightness: 106.46, 107.00, 119.42 (RGB)
- Standard deviation: 62.93, 62.97, 57.64
- Unique colors: 9,653
- Skin-tone pixels: 0 (0.00%)

**Key Finding:**
Frame 4 contains 34 pixels classified as "skin-tone" (R > 95, G > 40, B > 20, R > G > B), while frames 0 and 12 have ZERO skin-tone pixels.

## Hypothesis: Pose Orientation Matters

**Current hypothesis:** The actual POSE/ORIENTATION of the mannequin differs between frames:
- Frame 4 likely shows a pose where the mannequin has visible "flesh-colored" pixels
- Frames 0 and 12 likely show poses where the mannequin is completely covered or facing differently
- OpenPose may be failing on frames where it cannot detect human-like flesh tones OR where the body orientation is ambiguous

**The problem is NOT:**
- Image size/resolution
- Centering/positioning
- Brightness/contrast
- OpenPose configuration parameters

**The problem IS likely:**
- The actual VISUAL CONTENT of the sprite (pose orientation, visible body parts)
- Presence/absence of skin-tone colored pixels
- OpenPose's ability to recognize the body structure in that specific pose

## What This Means

The plan's assumption that "all three frames are back-facing" may be incorrect. The frames may show different orientations or the mannequin design may differ in ways that affect OpenPose's human pose detection algorithm.

Frame 4 has characteristics that OpenPose recognizes as a human pose (including skin-tone-like pixels), while frames 0 and 12 do not trigger OpenPose's detection algorithms despite being visually similar to the human eye.

## Next Steps

1. **Visual Inspection Required:** Manually examine frames 0, 4, and 12 to identify VISUAL differences in pose/orientation
2. **Test Hypothesis:** If frames differ in orientation, test if adding skin-tone pixels or adjusting pose affects detection
3. **Preprocessing Strategy:** Based on visual differences, determine if:
   - Frames can be normalized to match frame 4's characteristics
   - Alternative detection methods needed for certain poses
   - Manual skeleton annotation required for failing frames

## Data Files Generated

- `/Users/roberthyatt/Code/ComfyUI/analyze_openpose_frames.py` - Image property analysis script
- `/Users/roberthyatt/Code/ComfyUI/analyze_skeleton_outputs.py` - Skeleton output verification script
- `/Users/roberthyatt/Code/ComfyUI/deep_pixel_analysis.py` - Pixel-level color analysis script
- `/Users/roberthyatt/Code/ComfyUI/analyze_diagnostic_results.py` - Multi-configuration test analysis
- `/Users/roberthyatt/Code/ComfyUI/diagnostic_analysis.json` - Detailed test results
- `/Users/roberthyatt/Code/ComfyUI/output/openpose_diagnostic/` - Visual outputs from all test configurations
