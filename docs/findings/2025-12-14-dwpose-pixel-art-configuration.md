# DWPose Configuration for Pixel Art Sprites - December 14, 2025

## Status: WORKING - Requires Specific Configuration

## Critical Finding

**DWPose ONLY works on pixel art sprites when `bbox_detector="None"`**

### What Works
```python
"bbox_detector": "None",  # CRITICAL: Skip YOLOX for pixel art
"pose_estimator": "dw-ll_ucoco_384.onnx"
```

### What Does NOT Work
```python
"bbox_detector": "yolox_l.onnx",  # FAILS: Returns 0 keypoints
"pose_estimator": "dw-ll_ucoco_384.onnx"
```

## Why This Happens

1. **YOLOX is a person detector** trained on photographic images
2. **Pixel art sprites don't look like people to YOLOX** - no bounding box detected
3. **Without a bounding box, DWPose has nothing to process** → black output
4. **Setting `bbox_detector="None"` tells DWPose to process the entire image** instead of waiting for a person bounding box

## Test Results

| Frame | bbox_detector | Pixels Detected | Status |
|-------|--------------|-----------------|--------|
| base_frame_00.png | yolox_l.onnx | 0 | FAILED |
| base_frame_00.png | None | 6,269 | DETECTED |
| base_frame_05.png | None | 6,905 | DETECTED |
| base_frame_10.png | None | 5,740 | DETECTED |
| clothed_frame_00.png | None | 5,653 | DETECTED |
| clothed_frame_05.png | None | 5,931 | DETECTED |
| clothed_frame_10.png | None | 4,751 | DETECTED |

**100% detection rate with `bbox_detector="None"`**

## Full Working Workflow

```python
workflow = {
    "1": {
        "inputs": {"image": "sprite_frame.png"},
        "class_type": "LoadImage"
    },
    "2": {
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "bbox_detector": "None",        # <-- CRITICAL FOR PIXEL ART
            "pose_estimator": "dw-ll_ucoco_384.onnx",
            "image": ["1", 0]
        },
        "class_type": "DWPreprocessor"
    },
    "3": {
        "inputs": {
            "filename_prefix": "dwpose_output",
            "images": ["2", 0]
        },
        "class_type": "SaveImage"
    }
}
```

## Files Where This Is Already Configured Correctly

- `sprite_clothing_gen/workflow_builder.py:161` - Uses `"None"` with comment
- `sprite_clothing_gen/workflow_builder.py:375` - Uses `"None"` with comment

## Files That Need Fixing (If They Exist)

- `test_dwpose_comparison.py:41` - Uses `"yolox_l.onnx"` (will fail on pixel art)

## DWPose Limitation (Still Applies)

Even with skeleton detection working, DWPose has **dimensionality collapse** - it cannot distinguish front-facing from back-facing poses. The 2D keypoint positions are identical for both orientations.

See: `docs/findings/2025-12-13-depth-controlnet-orientation-fix.md` for details on this limitation.

## Key Takeaway

**When working with pixel art sprites in ComfyUI:**
1. Always use `bbox_detector="None"` for DWPreprocessor
2. This bypasses YOLOX person detection which fails on non-photorealistic images
3. DWPose will then process the entire image and successfully detect keypoints
