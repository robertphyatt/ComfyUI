# Golden Frame Color Correction Design

**Date:** 2025-12-18
**Status:** Ready for implementation

## Problem

The current pipeline processes each frame independently, causing pixel color inconsistencies between frames. This creates an "uncanny valley" effect in the animation where armor pixels have slightly different colors from frame to frame.

## Solution

After running the existing pipeline (without pixelation), the user visually selects a "golden frame." The system then color-corrects all other frames to use the exact pixel colors from the golden frame, mapped by body segment and keypoint-relative positions. Pixelation is applied as the final step.

## Workflow

Single command:
```bash
python -m sprite_keypoint_detector.pipeline \
  --frames-dir training_data/frames/ \
  --annotations training_data/annotations.json \
  --masks training_data/masks_corrected/ \
  --output /tmp/pipeline_test/ \
  --debug  # optional: show intermediate steps
```

Steps:
1. Run existing pipeline (scale/align → mask → rotate → inpaint) for all 25 frames - **no pixelation**
2. GUI popup: thumbnail grid + large preview, click through frames, click "Select as Golden"
3. Color correction: map all pixels to golden frame colors using body-segment keypoint mapping
4. Pixelation: apply pixelize step to all frames
5. Output final spritesheet (+ debug comparisons if `--debug`)

## Body Segment Mapping

Segments based on 18-point skeleton:

| Segment | Keypoints | Description |
|---------|-----------|-------------|
| Head | 0, 1 | Head/neck area |
| Torso | 1, 2, 5, 8, 11 | Main body |
| Left Upper Arm | 5, 6 | Left shoulder → elbow |
| Left Lower Arm | 6, 7 | Left elbow → wrist |
| Right Upper Arm | 2, 3 | Right shoulder → elbow |
| Right Lower Arm | 3, 4 | Right elbow → wrist |
| Left Upper Leg | 11, 12 | Left hip → knee |
| Left Lower Leg | 12, 13 | Left knee → ankle |
| Right Upper Leg | 8, 9 | Right hip → knee |
| Right Lower Leg | 9, 10 | Right knee → ankle |

**Pixel-to-Segment Assignment:**
- For each visible armor pixel, find which segment it belongs to
- Store its position relative to that segment's keypoints (e.g., "30% along the bone, 5px perpendicular left")

**Mapping Between Frames:**
- Given pixel P in frame X belonging to "Right Lower Arm"
- Compute P's relative position to keypoints 3→4 in frame X
- Find the same relative position using keypoints 3→4 in golden frame
- Copy that golden pixel's RGB value

## Color Correction Process

For each non-golden frame:

1. **Build golden frame lookup:**
   - For each visible pixel in golden frame, compute its segment and relative position
   - Store in spatial data structure for fast lookup

2. **For each pixel in target frame:**
   - If pixel is transparent, skip
   - Determine which segment it belongs to
   - Compute its relative position within that segment
   - Query golden frame for pixel at same relative position
   - If found: copy RGB from golden
   - If not found: find nearest neighbor in golden frame within same segment, copy that RGB

3. **Edge case - segment not visible in golden:**
   - Fall back to nearest visible segment or keep original color
   - Should be rare if user picks a good golden frame with full visibility

## GUI for Golden Frame Selection

**Implementation:** Matplotlib interactive figure

**Layout:**
- Left panel: 5x5 grid of thumbnail images (all 25 frames)
- Right panel: Large preview of currently selected frame
- Bottom: "Select as Golden" button, frame number label

**Interaction:**
- Click any thumbnail → shows large preview on right
- Click "Select as Golden" button → closes GUI, returns selected frame index
- Keyboard: arrow keys to navigate, Enter to select

## Pipeline Integration

**Changes to existing pipeline:**

1. Remove pixelation from main transform flow:
   - `transform_frame()` returns after inpainting (no pixelize call)
   - `apply_pixelize()` remains available as standalone function

2. New pipeline stages:
   ```
   Existing:  align → mask → rotate → inpaint
   New:       → [GUI select golden] → color_correct → pixelize
   ```

3. New functions:
   - `select_golden_frame(frames)` - GUI, returns index
   - `assign_pixel_segments(image, keypoints)` - returns segment ID per pixel
   - `compute_relative_positions(image, keypoints, segments)` - returns position data
   - `color_correct_frame(frame, frame_kpts, golden, golden_kpts)` - returns corrected frame
   - `color_correct_all(frames, keypoints, golden_idx)` - orchestrates correction

4. Debug output (with --debug flag):
   - Existing: `1_aligned/`, `2_masked/`, `3_rotated/`, `4_inpainted/`
   - New: `5_color_corrected/`, `6_final/`, `comparison/`

## Files

**Modify:**
- `sprite_keypoint_detector/transform.py` - remove pixelize from `transform_frame()`, keep `apply_pixelize()` standalone
- `sprite_keypoint_detector/pipeline.py` - integrate new stages, update debug output

**Create:**
- `sprite_keypoint_detector/golden_selection.py` - GUI for selecting golden frame
- `sprite_keypoint_detector/color_correction.py` - segment assignment, relative position mapping, color correction logic

## Estimated Scope

~300-400 lines of new code
