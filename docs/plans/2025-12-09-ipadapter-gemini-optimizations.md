# IPAdapter Clothing Transfer Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Systematically test Gemini's research suggestions to improve IPAdapter clothing generation quality and consistency.

**Architecture:** Incremental testing approach - test each optimization (ControlNet Tile, img2img with denoise=0.65, AnimateDiff) independently, then combine the successful ones.

**Tech Stack:** ComfyUI, IPAdapter Plus, ControlNet (Tile and OpenPose variants), Python workflow builder

---

## Current Working Baseline

**What works:**
- IPAdapter + ControlNet OpenPose + txt2img (EmptyLatentImage)
- Generates character with brown leather armor
- Correct pose from ControlNet

**Issues to solve:**
- Potential flickering between frames (no temporal consistency)
- Background changes between frames
- May benefit from ControlNet Tile for pixel art

---

## Task 1: Test ControlNet Tile vs OpenPose

**Rationale:** Gemini research (Section 3.2.2) argues ControlNet Tile is superior for pixel art sprites because OpenPose fails on few-pixel-wide limbs.

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/test_controlnet_tile.py`
- Reference: `/Users/roberthyatt/Code/ComfyUI/sprite_clothing_gen/workflow_builder.py`

### Step 1: Check if ControlNet Tile model is available

```bash
ls -la /Users/roberthyatt/Code/ComfyUI/models/controlnet/ | grep -i tile
```

**Expected:** Find `control_v11f1e_sd15_tile.pth` or similar

**If missing:** Download from https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11f1e_sd15_tile.pth

### Step 2: Create test script comparing Tile vs OpenPose

**File:** `/Users/roberthyatt/Code/ComfyUI/test_controlnet_tile.py`

```python
"""Compare ControlNet Tile vs OpenPose for pixel art sprite generation."""
import sys
from pathlib import Path
from sprite_clothing_gen.comfy_client import ComfyUIClient

def build_test_workflow_tile(reference_image: str, base_image: str, seed: int = 42):
    """IPAdapter + ControlNet TILE (no OpenPose preprocessing)."""
    workflow = {
        # Checkpoint
        "1": {
            "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },

        # IPAdapter Unified Loader
        "2": {
            "inputs": {
                "model": ["1", 0],
                "preset": "PLUS (high strength)"
            },
            "class_type": "IPAdapterUnifiedLoader"
        },

        # Load reference image (clothed)
        "3": {
            "inputs": {"image": reference_image},
            "class_type": "LoadImage"
        },

        # Load base image (for ControlNet)
        "4": {
            "inputs": {"image": base_image},
            "class_type": "LoadImage"
        },

        # ControlNet Loader - TILE variant
        "5": {
            "inputs": {"control_net_name": "control_v11f1e_sd15_tile.pth"},
            "class_type": "ControlNetLoader"
        },

        # Apply IPAdapter
        "7": {
            "inputs": {
                "weight": 1.0,
                "weight_type": "style and composition",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "ipadapter": ["2", 1],
                "image": ["3", 0],
                "model": ["2", 0]
            },
            "class_type": "IPAdapterAdvanced"
        },

        # Positive prompt
        "8": {
            "inputs": {
                "text": "character wearing brown leather armor, pixel art",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },

        # Negative prompt
        "9": {
            "inputs": {
                "text": "blurry, low quality",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },

        # Apply ControlNet TILE - uses base image directly (no OpenPose)
        "10": {
            "inputs": {
                "strength": 0.7,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "positive": ["8", 0],
                "negative": ["9", 0],
                "control_net": ["5", 0],
                "image": ["4", 0]  # Direct base image, no preprocessing
            },
            "class_type": "ControlNetApplyAdvanced"
        },

        # Empty latent (txt2img)
        "11": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },

        # KSampler
        "12": {
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["7", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["11", 0]
            },
            "class_type": "KSampler"
        },

        # VAE Decode
        "13": {
            "inputs": {
                "samples": ["12", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },

        # Save
        "14": {
            "inputs": {
                "filename_prefix": "test_controlnet_tile",
                "images": ["13", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow

# Test
client = ComfyUIClient("http://127.0.0.1:8188")

if not client.health_check():
    print("ERROR: ComfyUI not running")
    sys.exit(1)

print("Uploading images...")
client.upload_image(Path("input/clothed_frame_00.png"))
client.upload_image(Path("training_data/frames/base_frame_00.png"))

print("Testing ControlNet TILE...")
workflow = build_test_workflow_tile("clothed_frame_00.png", "base_frame_00.png", seed=777)

prompt_id = client.queue_prompt(workflow)
history = client.wait_for_completion(prompt_id, timeout=120)

print("\n✓ Test complete - check output/test_controlnet_tile_*.png")
print("\nCompare against output/ipadapter_test_with_controlnet_*.png (OpenPose version)")
print("Visual check:")
print("  - Does Tile version preserve exact mannequin silhouette better?")
print("  - Does Tile version have cleaner pixel edges?")
print("  - Does armor fit the mannequin shape more accurately?")
```

### Step 3: Run ControlNet Tile test

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 test_controlnet_tile.py
```

**Expected output:** `output/test_controlnet_tile_*.png` with armored character

### Step 4: Visual comparison

**Compare side-by-side:**
- `output/test_controlnet_tile_*.png` (new)
- `output/ipadapter_test_with_controlnet_*.png` (OpenPose baseline)

**Evaluation criteria:**
- Silhouette accuracy: Does Tile preserve the exact mannequin outline?
- Pixel sharpness: Are edges cleaner with Tile?
- Armor fit: Does armor conform better to mannequin shape?

### Step 5: Document findings

Create: `/Users/roberthyatt/Code/ComfyUI/docs/findings/2025-12-09-controlnet-tile-vs-openpose.md`

```markdown
# ControlNet Tile vs OpenPose Comparison

**Date:** 2025-12-09
**Test:** ControlNet Tile vs OpenPose for pixel art sprite generation

## Results

**ControlNet Tile:**
- [Visual assessment]
- [Silhouette accuracy]
- [Edge quality]

**ControlNet OpenPose:**
- [Visual assessment]
- [Silhouette accuracy]
- [Edge quality]

## Recommendation

[Choose winner based on visual quality]
```

---

## Task 2: Test img2img with denoise=0.65

**Rationale:** Gemini research (Section 4.4) suggests img2img with denoise=0.65 preserves motion flow while replacing skin with armor.

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/test_img2img_denoise.py`

### Step 1: Create img2img test script

**File:** `/Users/roberthyatt/Code/ComfyUI/test_img2img_denoise.py`

```python
"""Test img2img with denoise=0.65 vs txt2img."""
import sys
from pathlib import Path
from sprite_clothing_gen.comfy_client import ComfyUIClient

def build_img2img_workflow(reference_image: str, base_image: str, denoise: float, seed: int = 42):
    """IPAdapter + ControlNet + img2img with configurable denoise."""
    workflow = {
        # Checkpoint
        "1": {
            "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },

        # IPAdapter Unified Loader
        "2": {
            "inputs": {
                "model": ["1", 0],
                "preset": "PLUS (high strength)"
            },
            "class_type": "IPAdapterUnifiedLoader"
        },

        # Load reference image (clothed)
        "3": {
            "inputs": {"image": reference_image},
            "class_type": "LoadImage"
        },

        # Load base image
        "4": {
            "inputs": {"image": base_image},
            "class_type": "LoadImage"
        },

        # ControlNet Loader (use winner from Task 1)
        "5": {
            "inputs": {"control_net_name": "control_v11p_sd15_openpose_fp16.safetensors"},
            "class_type": "ControlNetLoader"
        },

        # OpenPose Preprocessor
        "6": {
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "image": ["4", 0]
            },
            "class_type": "OpenposePreprocessor"
        },

        # Apply IPAdapter
        "7": {
            "inputs": {
                "weight": 1.0,
                "weight_type": "style and composition",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "ipadapter": ["2", 1],
                "image": ["3", 0],
                "model": ["2", 0]
            },
            "class_type": "IPAdapterAdvanced"
        },

        # Positive prompt
        "8": {
            "inputs": {
                "text": "character wearing brown leather armor, pixel art",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },

        # Negative prompt
        "9": {
            "inputs": {
                "text": "blurry, low quality",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },

        # Apply ControlNet
        "10": {
            "inputs": {
                "strength": 0.7,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "positive": ["8", 0],
                "negative": ["9", 0],
                "control_net": ["5", 0],
                "image": ["6", 0]
            },
            "class_type": "ControlNetApplyAdvanced"
        },

        # VAE Encode base image (img2img starting point)
        "11": {
            "inputs": {
                "pixels": ["4", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEEncode"
        },

        # KSampler with configurable denoise
        "12": {
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,  # The key parameter!
                "model": ["7", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["11", 0]  # From VAEEncode, not EmptyLatentImage
            },
            "class_type": "KSampler"
        },

        # VAE Decode
        "13": {
            "inputs": {
                "samples": ["12", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },

        # Save
        "14": {
            "inputs": {
                "filename_prefix": f"test_img2img_denoise_{int(denoise*100)}",
                "images": ["13", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow

# Test multiple denoise values
client = ComfyUIClient("http://127.0.0.1:8188")

if not client.health_check():
    print("ERROR: ComfyUI not running")
    sys.exit(1)

print("Uploading images...")
client.upload_image(Path("input/clothed_frame_00.png"))
client.upload_image(Path("training_data/frames/base_frame_00.png"))

denoise_values = [0.50, 0.65, 0.80, 1.0]

for denoise in denoise_values:
    print(f"\nTesting denoise={denoise}...")
    workflow = build_img2img_workflow(
        "clothed_frame_00.png",
        "base_frame_00.png",
        denoise=denoise,
        seed=666
    )

    prompt_id = client.queue_prompt(workflow)
    history = client.wait_for_completion(prompt_id, timeout=120)
    print(f"  ✓ Saved as test_img2img_denoise_{int(denoise*100)}_*.png")

print("\n✓ All tests complete")
print("\nCompare outputs:")
print("  - denoise=0.50: More mannequin preserved")
print("  - denoise=0.65: Gemini's recommended balance")
print("  - denoise=0.80: More generation freedom")
print("  - denoise=1.00: Full regeneration (current txt2img equivalent)")
```

### Step 2: Run img2img denoise tests

```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 test_img2img_denoise.py
```

**Expected:** 4 output files with different denoise levels

### Step 3: Visual comparison

**Compare:**
- `test_img2img_denoise_50_*.png`
- `test_img2img_denoise_65_*.png` (Gemini recommendation)
- `test_img2img_denoise_80_*.png`
- `test_img2img_denoise_100_*.png`
- Current baseline (txt2img)

**Evaluation criteria:**
- Armor presence and quality
- Mannequin silhouette preservation
- Background consistency
- Detail sharpness

### Step 4: Document findings

Create: `/Users/roberthyatt/Code/ComfyUI/docs/findings/2025-12-09-img2img-denoise-comparison.md`

```markdown
# img2img Denoise Level Comparison

**Date:** 2025-12-09
**Test:** img2img with varying denoise vs txt2img baseline

## Results

**denoise=0.50:**
- [Assessment]

**denoise=0.65 (Gemini recommendation):**
- [Assessment]

**denoise=0.80:**
- [Assessment]

**denoise=1.00:**
- [Assessment]

**txt2img baseline:**
- [Assessment]

## Recommendation

[Choose optimal denoise value or stick with txt2img]
```

---

## Task 3: Update Production Workflow with Winner

**Goal:** Apply the best combination from Tasks 1-2 to the production workflow.

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_clothing_gen/workflow_builder.py`

### Step 1: Identify winning configuration

Based on Tasks 1-2 findings, determine:
- ✅ ControlNet type: [Tile OR OpenPose]
- ✅ Latent source: [img2img with denoise=X OR txt2img]

### Step 2: Update workflow_builder.py

**Example for ControlNet Tile winner:**

```python
# In build_ipadapter_generation_workflow()

# Replace OpenPose preprocessor with direct base image for Tile
workflow[controlnet_loader_id] = {
    "inputs": {"control_net_name": "control_v11f1e_sd15_tile.pth"},
    "class_type": "ControlNetLoader",
    "_meta": {"title": "Load ControlNet Tile"}
}

# Remove OpenPose preprocessor node entirely
# Apply ControlNet directly to base image
workflow[controlnet_apply_id] = {
    "inputs": {
        "strength": 0.7,
        "start_percent": 0.0,
        "end_percent": 1.0,
        "positive": [clip_positive_id, 0],
        "negative": [clip_negative_id, 0],
        "control_net": [controlnet_loader_id, 0],
        "image": ["1", 0]  # Direct base image, no preprocessing
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {"title": "Apply ControlNet Tile"}
}
```

**Example for img2img with denoise=0.65 winner:**

```python
# Replace EmptyLatentImage with VAEEncode
workflow[vae_encode_id] = {
    "inputs": {
        "pixels": ["1", 0],  # Base image
        "vae": [checkpoint_node_id, 2]
    },
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode (img2img)"}
}

# Update KSampler with denoise
workflow[ksampler_id] = {
    "inputs": {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.65,  # Gemini's recommended value
        "model": [ipadapter_apply_id, 0],
        "positive": [controlnet_apply_id, 0],
        "negative": [controlnet_apply_id, 1],
        "latent_image": [vae_encode_id, 0]  # From VAEEncode
    },
    "class_type": "KSampler",
    "_meta": {"title": "KSampler (img2img)"}
}
```

### Step 3: Test updated production workflow

```bash
cd /Users/roberthyatt/Code/ComfyUI
rm -f training_data/frames_ipadapter_generated/clothed_frame_00.png
python3 generate_with_ipadapter.py 2>&1 | grep -A5 "Frame 00"
```

**Expected:** Frame 00 generates successfully with improvements from chosen configuration

### Step 4: Visual verification

```bash
# View the generated frame
open training_data/frames_ipadapter_generated/clothed_frame_00.png
```

**Check:**
- Brown leather armor present ✓
- Correct pose ✓
- Improved quality vs baseline

### Step 5: Commit changes

```bash
git add sprite_clothing_gen/workflow_builder.py
git add docs/findings/2025-12-09-*.md
git commit -m "feat: apply Gemini research optimizations to IPAdapter workflow

- [Tile/OpenPose]: [reasoning]
- [img2img/txt2img]: [reasoning]
- Denoise: [value if img2img]

Visual quality improvements confirmed in test frames.
"
```

---

## Task 4: OPTIONAL - AnimateDiff Temporal Consistency

**Note:** This is an advanced optimization. Only pursue if Tasks 1-3 show flickering between frames.

**Rationale:** Gemini research (Section 6.1) suggests AnimateDiff for frame-to-frame consistency.

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/test_animatediff.py`

### Step 1: Check AnimateDiff availability

```bash
ls -la /Users/roberthyatt/Code/ComfyUI/custom_nodes/ | grep -i animate
```

**Expected:** AnimateDiff extension installed

**If missing:**
```bash
cd /Users/roberthyatt/Code/ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
# Restart ComfyUI
```

### Step 2: Download AnimateDiff motion module

```bash
cd /Users/roberthyatt/Code/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved
# Follow their download instructions for mm_sd_v15_v2.ckpt
```

### Step 3: Create AnimateDiff test workflow

**File:** `/Users/roberthyatt/Code/ComfyUI/test_animatediff.py`

```python
"""Test AnimateDiff for temporal consistency across sprite frames."""
# Full workflow with AnimateDiff node insertion
# [Implementation would go here based on AnimateDiff documentation]
```

### Step 4: Test with and without AnimateDiff

Generate 3 consecutive frames with each approach and compare for flickering.

### Step 5: Document findings

Only implement if flickering is observed in multi-frame generation.

---

## Success Criteria

**After completing this plan:**

✅ Systematic comparison of ControlNet Tile vs OpenPose
✅ Systematic comparison of img2img denoise levels vs txt2img
✅ Production workflow updated with best configuration
✅ All findings documented with visual evidence
✅ Single-frame generation produces high-quality armored character
✅ Ready for 25-frame batch generation

**Next steps after this plan:**
- Run full 25-frame generation
- Visual inspection for consistency
- If flickering detected, pursue Task 4 (AnimateDiff)
