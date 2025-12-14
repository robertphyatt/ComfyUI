# Task 3: Production Workflow Update - Verification & Testing

**Date:** 2025-12-10
**Task:** Verify and update production workflow with winning configuration from Tasks 1-2
**Status:** ✅ Complete

---

## Summary

Verified that the production workflow in `sprite_clothing_gen/workflow_builder.py` required updates to match the winning configuration from Tasks 1-2. Successfully updated the workflow and confirmed it generates frames correctly.

---

## Winning Configuration (from Tasks 1-2)

- **ControlNet:** OpenPose (`control_v11p_sd15_openpose_fp16.safetensors`)
- **Latent Source:** EmptyLatentImage (txt2img approach)
- **Denoise:** 1.0

---

## Production Workflow Status

### Initial State (BEFORE Changes)

**Configuration:**
- ✅ ControlNet: OpenPose - **CORRECT**
- ❌ Latent source: VAEEncode (img2img) - **INCORRECT**
- ❌ Used SetLatentNoiseMask for inpainting - **INCORRECT**

**Analysis:**
The production workflow had the correct ControlNet (OpenPose) but was using the img2img approach (VAEEncode + SetLatentNoiseMask) instead of the winning txt2img approach.

### Changes Made

**File:** `/Users/roberthyatt/Code/ComfyUI/sprite_clothing_gen/workflow_builder.py`

**1. Replaced VAEEncode with EmptyLatentImage (lines 418-430)**

Before:
```python
workflow[vae_encode_id] = {
    "inputs": {
        "pixels": ["1", 0],
        "vae": [checkpoint_node_id, 2]
    },
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
}

workflow[set_mask_id] = {
    "inputs": {
        "samples": [vae_encode_id, 0],
        "mask": ["2", 1]
    },
    "class_type": "SetLatentNoiseMask",
    "_meta": {"title": "Set Latent Noise Mask"}
}
```

After:
```python
empty_latent_id = vae_encode_id  # Reuse variable name for cleaner refactor
workflow[empty_latent_id] = {
    "inputs": {
        "width": 512,
        "height": 512,
        "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {"title": "Empty Latent Image"}
}

# Note: SetLatentNoiseMask removed - using txt2img (denoise=1.0) instead of img2img
# KSampler will use EmptyLatentImage directly
```

**2. Updated KSampler to use EmptyLatentImage (line 443)**

Before:
```python
"latent_image": [set_mask_id, 0]
```

After:
```python
"latent_image": [empty_latent_id, 0]  # Changed from set_mask_id to empty_latent_id
```

**3. Removed set_mask_id variable declaration (line 339)**

Before:
```python
vae_encode_id = str(batch_node_id)
batch_node_id += 1
set_mask_id = str(batch_node_id)
batch_node_id += 1
ksampler_id = str(batch_node_id)
```

After:
```python
vae_encode_id = str(batch_node_id)  # Will be renamed to empty_latent_id below
batch_node_id += 1
# set_mask_id removed - no longer using SetLatentNoiseMask (txt2img instead of img2img)
ksampler_id = str(batch_node_id)
```

**4. Fixed checkpoint_node_id reference bug (lines 203-211)**

Moved `checkpoint_node_id` calculation BEFORE the workflow dict initialization to fix `UnboundLocalError`. This was a pre-existing bug that prevented the workflow from being built.

Before:
```python
workflow = {
    # ...
    "3": {
        "inputs": {
            "model": [checkpoint_node_id, 0],  # ← UnboundLocalError!
```

After:
```python
# Calculate checkpoint node ID FIRST
checkpoint_node_id = str(29 + max_batch_nodes)

workflow = {
    # ...
    "3": {
        "inputs": {
            "model": [checkpoint_node_id, 0],  # ✓ Now defined
```

---

## Verification

### Workflow Structure Tests

Created `/Users/roberthyatt/Code/ComfyUI/test_task3_workflow_verification.py` to verify:

- ✅ ControlNet: `control_v11p_sd15_openpose_fp16.safetensors`
- ✅ OpenposePreprocessor present
- ✅ Using EmptyLatentImage (txt2img)
- ✅ NOT using VAEEncode (img2img)
- ✅ NOT using SetLatentNoiseMask
- ✅ KSampler denoise: 1.0
- ✅ KSampler latent_image source: EmptyLatentImage

**Result:** All tests passed ✅

### Generation Test

**Test:** Generate Frame 00 with updated workflow

**Command:**
```bash
cd /Users/roberthyatt/Code/ComfyUI
rm -f training_data/frames_ipadapter_generated/clothed_frame_00.png
python3 generate_with_ipadapter.py
```

**Result:**
```
Frame 00:
  Uploading base and mask...
  Building workflow...
  Generating with IPAdapter...
  ✓ Saved to training_data/frames_ipadapter_generated/clothed_frame_00.png
```

**Output File:**
- Path: `training_data/frames_ipadapter_generated/clothed_frame_00.png`
- Size: 180KB
- Status: ✅ Generated successfully

---

## Configuration Summary

### Final Production Workflow Configuration

```python
# ControlNet
ControlNetLoader: "control_v11p_sd15_openpose_fp16.safetensors"
OpenposePreprocessor: enabled (detect_hand, detect_body, detect_face)

# Latent Source (txt2img approach)
EmptyLatentImage: 512x512, batch_size=1

# KSampler
denoise: 1.0 (full generation, not img2img refinement)
latent_image: from EmptyLatentImage
```

### Removed Components

- ❌ VAEEncode (img2img starting point)
- ❌ SetLatentNoiseMask (inpainting mask)

### Rationale

The txt2img approach (EmptyLatentImage + denoise=1.0) won over img2img in Task 2 testing because:
1. Full generation freedom (not constrained by base image pixels)
2. Better integration of IPAdapter style transfer
3. OpenPose provides sufficient pose guidance without pixel-level constraints

---

## Files Changed

1. `/Users/roberthyatt/Code/ComfyUI/sprite_clothing_gen/workflow_builder.py`
   - Replaced VAEEncode with EmptyLatentImage
   - Removed SetLatentNoiseMask
   - Fixed checkpoint_node_id reference bug
   - Updated KSampler to use EmptyLatentImage

2. `/Users/roberthyatt/Code/ComfyUI/test_task3_workflow_verification.py` (NEW)
   - Workflow structure verification tests

3. `/Users/roberthyatt/Code/ComfyUI/docs/findings/2025-12-10-task3-production-workflow-update.md` (NEW)
   - This document

---

## Next Steps

1. ✅ Production workflow updated with winning configuration
2. ✅ Workflow verified with automated tests
3. ✅ Frame 00 generation test successful
4. ⏭️ Commit changes
5. ⏭️ Consider Task 4 (AnimateDiff) if temporal consistency issues observed

---

## Notes

- The production workflow was using img2img before this update, which may explain any previous generation quality issues
- The txt2img approach eliminates the need for inpainting masks (SetLatentNoiseMask), simplifying the workflow
- OpenPose provides pose guidance while EmptyLatentImage gives IPAdapter full freedom for style transfer
- The checkpoint_node_id bug fix was necessary for the workflow to run at all - this suggests the workflow may not have been tested recently
