# IPAdapter + ControlNet Clothing Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate clothing in exact base poses using IPAdapter (learning from 25 reference frames) + OpenPose ControlNet, then extract using existing masking system.

**Architecture:** ComfyUI workflow with IPAdapter (batch references), OpenPose ControlNet (pose control), and inpainting (masked generation). Generated frames run through existing U-Net masking pipeline for clothing extraction.

**Tech Stack:** ComfyUI, IPAdapter, ControlNet, OpenPose, Python, PIL, NumPy

---

## Prerequisites

- ComfyUI server running at http://127.0.0.1:8188
- IPAdapter model installed in ComfyUI
- OpenPose ControlNet model installed in ComfyUI
- Existing trained U-Net masking model (from previous work)
- 25 base frames in `training_data/frames/base_frame_XX.png`
- 25 clothed reference frames in `training_data/frames/clothed_frame_XX.png`

---

### Task 1: Create Mask Generator from Alpha Channel

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/generate_inpainting_masks.py`
- Test: `/Users/roberthyatt/Code/ComfyUI/tests/test_mask_generation.py`

**Step 1: Write the failing test**

Create test file:
```python
"""Tests for inpainting mask generation."""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from generate_inpainting_masks import generate_mask_from_alpha


def test_mask_from_alpha_creates_binary_mask():
    """Test that alpha channel creates proper binary mask."""
    # Create test RGBA image (100x100 with center square opaque)
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[25:75, 25:75, 3] = 255  # Opaque square in center

    test_img = Image.fromarray(img, 'RGBA')

    # Generate mask
    mask = generate_mask_from_alpha(test_img)

    # Verify mask is binary (0 or 255)
    assert mask.mode == 'L'
    assert mask.size == (100, 100)

    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)
    assert len(unique_values) <= 2
    assert 0 in unique_values
    assert 255 in unique_values

    # Verify mask matches alpha channel
    assert np.all(mask_array[25:75, 25:75] == 255)  # Center is white
    assert np.all(mask_array[0:25, :] == 0)  # Top is black


def test_mask_generation_for_all_base_frames():
    """Test generating masks for actual base frames."""
    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/masks_inpainting")

    base_path = frames_dir / "base_frame_00.png"
    if not base_path.exists():
        pytest.skip("Base frame not found")

    from generate_inpainting_masks import generate_masks_for_frames

    # Generate for frame 0 only
    generate_masks_for_frames(frames_dir, output_dir, frame_range=(0, 1))

    # Verify output exists
    mask_path = output_dir / "mask_00.png"
    assert mask_path.exists()

    # Load and verify
    mask = Image.open(mask_path)
    assert mask.mode == 'L'
    assert mask.size == (512, 512)
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
pytest tests/test_mask_generation.py -v
```

Expected: `ModuleNotFoundError: No module named 'generate_inpainting_masks'`

**Step 3: Write minimal implementation**

Create implementation file:
```python
#!/usr/bin/env python3
"""Generate inpainting masks from alpha channel of base frames."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple


def generate_mask_from_alpha(image: Image.Image) -> Image.Image:
    """Generate binary inpainting mask from alpha channel.

    Args:
        image: RGBA image

    Returns:
        Binary mask (L mode): white = generate here, black = keep original
    """
    # Extract alpha channel
    if image.mode != 'RGBA':
        raise ValueError(f"Image must be RGBA, got {image.mode}")

    alpha = np.array(image)[:, :, 3]

    # Create binary mask: alpha > 0 → white (255), else black (0)
    mask = (alpha > 0).astype(np.uint8) * 255

    return Image.fromarray(mask, mode='L')


def generate_masks_for_frames(frames_dir: Path, output_dir: Path,
                              frame_range: Tuple[int, int] = (0, 25)) -> None:
    """Generate inpainting masks for base frames.

    Args:
        frames_dir: Directory containing base_frame_XX.png files
        output_dir: Directory to save mask_XX.png files
        frame_range: (start, end) frame indices (end exclusive)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    start, end = frame_range

    for frame_idx in range(start, end):
        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"

        if not base_path.exists():
            print(f"Warning: {base_path} not found, skipping")
            continue

        # Load base frame
        base_img = Image.open(base_path).convert('RGBA')

        # Generate mask
        mask = generate_mask_from_alpha(base_img)

        # Save
        output_path = output_dir / f"mask_{frame_idx:02d}.png"
        mask.save(output_path)
        print(f"Generated {output_path}")


def main():
    """Generate inpainting masks for all base frames."""
    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/masks_inpainting")

    print("=" * 70)
    print("GENERATING INPAINTING MASKS FROM ALPHA CHANNEL")
    print("=" * 70)
    print()

    generate_masks_for_frames(frames_dir, output_dir)

    print()
    print("=" * 70)
    print("✓ All masks generated")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_mask_generation.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add generate_inpainting_masks.py tests/test_mask_generation.py
git commit -m "feat: add inpainting mask generation from alpha channel

Generates binary masks for inpainting workflow:
- White pixels = generate clothing here
- Black pixels = preserve background
- Simple alpha threshold (alpha > 0)"
```

---

### Task 2: Build IPAdapter + ControlNet Workflow

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_clothing_gen/workflow_builder.py`
- Test: `/Users/roberthyatt/Code/ComfyUI/tests/test_ipadapter_workflow.py`

**Step 1: Write the failing test**

```python
"""Tests for IPAdapter workflow builder."""
import pytest
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow


def test_workflow_has_required_nodes():
    """Test that workflow includes all required nodes."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        prompt="Brown leather armor",
        negative_prompt="blurry",
        seed=12345
    )

    # Verify workflow structure
    assert isinstance(workflow, dict)

    # Check for required node types
    node_classes = [node.get("class_type") for node in workflow.values()]

    required_nodes = [
        "LoadImage",  # Base image
        "LoadImage",  # Mask
        "IPAdapterModelLoader",
        "IPAdapterApply",
        "ControlNetLoader",
        "ControlNetApplyAdvanced",
        "CLIPTextEncode",  # Prompt
        "KSampler",  # Inpainting sampler
        "VAEDecode",
        "SaveImage"
    ]

    # Note: Can't check exact counts due to multiple LoadImage nodes
    # Just verify key node types exist
    assert "IPAdapterModelLoader" in node_classes
    assert "IPAdapterApply" in node_classes
    assert "ControlNetLoader" in node_classes
    assert "KSampler" in node_classes
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_has_required_nodes -v
```

Expected: `AttributeError: module 'sprite_clothing_gen.workflow_builder' has no attribute 'build_ipadapter_generation_workflow'`

**Step 3: Write minimal implementation**

Add to `sprite_clothing_gen/workflow_builder.py`:

```python
def build_ipadapter_generation_workflow(
    base_image_name: str,
    mask_image_name: str,
    reference_image_names: list[str],
    prompt: str = "Brown leather armor, pixel art",
    negative_prompt: str = "blurry, low quality",
    seed: int = 12345,
    steps: int = 35,
    cfg: float = 7.0,
    denoise: float = 1.0
) -> dict:
    """Build ComfyUI workflow for IPAdapter + ControlNet inpainting.

    Args:
        base_image_name: Filename of base character image
        mask_image_name: Filename of inpainting mask
        reference_image_names: List of 25 clothed reference frame filenames
        prompt: Positive text prompt
        negative_prompt: Negative text prompt
        seed: Random seed for reproducibility
        steps: Sampling steps
        cfg: CFG scale
        denoise: Denoise strength (1.0 = full generation)

    Returns:
        ComfyUI workflow dict
    """
    workflow = {
        # 1. Load base image
        "1": {
            "inputs": {"image": base_image_name},
            "class_type": "LoadImage"
        },

        # 2. Load inpainting mask
        "2": {
            "inputs": {"image": mask_image_name},
            "class_type": "LoadImage"
        },

        # 3. Load IPAdapter model
        "3": {
            "inputs": {"ipadapter_file": "ip-adapter_sd15.bin"},
            "class_type": "IPAdapterModelLoader"
        },

        # 4-28: Load 25 reference images (clothed frames)
        # For brevity, we'll load them in a batch node
        "4": {
            "inputs": {
                "mode": "incremental_image",
                "index": 0,
                "label": "batch",
                "path": "input/",
                "pattern": "clothed_frame_*.png",
                "allow_RGBA_output": "false"
            },
            "class_type": "LoadImageBatch"
        },

        # 29. Apply IPAdapter
        "29": {
            "inputs": {
                "weight": 0.8,
                "weight_type": "linear",
                "start_at": 0.0,
                "end_at": 1.0,
                "unfold_batch": "false",
                "ipadapter": ["3", 0],
                "image": ["4", 0],  # Reference images
                "model": ["30", 0]  # Will connect to checkpoint loader
            },
            "class_type": "IPAdapterApply"
        },

        # 30. Load checkpoint
        "30": {
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },

        # 31. Load ControlNet (OpenPose)
        "31": {
            "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"},
            "class_type": "ControlNetLoader"
        },

        # 32. OpenPose Preprocessor (extract skeleton from base)
        "32": {
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "image": ["1", 0]  # Base image
            },
            "class_type": "OpenposePreprocessor"
        },

        # 33. Apply ControlNet
        "33": {
            "inputs": {
                "strength": 0.9,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "positive": ["34", 0],  # Will connect to CLIP
                "negative": ["35", 0],
                "control_net": ["31", 0],
                "image": ["32", 0]  # OpenPose skeleton
            },
            "class_type": "ControlNetApplyAdvanced"
        },

        # 34. CLIP Text Encode (positive prompt)
        "34": {
            "inputs": {
                "text": prompt,
                "clip": ["30", 1]  # CLIP from checkpoint
            },
            "class_type": "CLIPTextEncode"
        },

        # 35. CLIP Text Encode (negative prompt)
        "35": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["30", 1]
            },
            "class_type": "CLIPTextEncode"
        },

        # 36. VAE Encode base image
        "36": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["30", 2]
            },
            "class_type": "VAEEncode"
        },

        # 37. Set Latent Noise Mask (for inpainting)
        "37": {
            "inputs": {
                "samples": ["36", 0],  # Latent from base
                "mask": ["2", 1]  # Mask image (alpha channel)
            },
            "class_type": "SetLatentNoiseMask"
        },

        # 38. KSampler (inpainting)
        "38": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": ["29", 0],  # IPAdapter model
                "positive": ["33", 0],  # ControlNet conditioning
                "negative": ["33", 1],
                "latent_image": ["37", 0]  # Masked latent
            },
            "class_type": "KSampler"
        },

        # 39. VAE Decode
        "39": {
            "inputs": {
                "samples": ["38", 0],
                "vae": ["30", 2]
            },
            "class_type": "VAEDecode"
        },

        # 40. Save Image
        "40": {
            "inputs": {
                "filename_prefix": "ipadapter_generated",
                "images": ["39", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_has_required_nodes -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add sprite_clothing_gen/workflow_builder.py tests/test_ipadapter_workflow.py
git commit -m "feat: add IPAdapter + ControlNet workflow builder

Creates ComfyUI workflow for:
- IPAdapter learning from 25 reference frames
- OpenPose ControlNet for pose control
- Inpainting for masked generation
- Generates clothed character in base pose"
```

---

### Task 3: Create IPAdapter Generation Script

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/generate_with_ipadapter.py`

**Step 1: Write implementation**

```python
#!/usr/bin/env python3
"""Generate clothed frames using IPAdapter + ControlNet."""

import sys
from pathlib import Path
from PIL import Image
from sprite_clothing_gen.comfy_client import ComfyUIClient
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow


def generate_clothed_frame(client: ComfyUIClient, frame_idx: int,
                          frames_dir: Path, masks_dir: Path,
                          output_dir: Path) -> bool:
    """Generate one clothed frame using IPAdapter.

    Args:
        client: ComfyUI client
        frame_idx: Frame index (0-24)
        frames_dir: Directory with base frames
        masks_dir: Directory with inpainting masks
        output_dir: Directory for generated frames

    Returns:
        True if successful
    """
    base_name = f"base_frame_{frame_idx:02d}.png"
    mask_name = f"mask_{frame_idx:02d}.png"

    # Upload base image and mask
    base_path = frames_dir / base_name
    mask_path = masks_dir / mask_name

    if not base_path.exists():
        print(f"  ✗ Base frame not found: {base_path}")
        return False

    if not mask_path.exists():
        print(f"  ✗ Mask not found: {mask_path}")
        return False

    print(f"  Uploading base and mask...")
    client.upload_image(base_path)
    client.upload_image(mask_path)

    # Build workflow
    print(f"  Building workflow...")

    # Reference images (all 25 clothed frames)
    reference_names = [f"clothed_frame_{i:02d}.png" for i in range(25)]

    workflow = build_ipadapter_generation_workflow(
        base_image_name=base_name,
        mask_image_name=mask_name,
        reference_image_names=reference_names,
        prompt="Brown leather armor with shoulder pauldrons, chest plate, arm guards, leg armor, fantasy RPG character, pixel art style, detailed, high quality",
        negative_prompt="blurry, low quality, distorted, deformed, multiple heads, extra limbs, modern clothing",
        seed=42 + frame_idx,  # Different seed per frame
        steps=35,
        cfg=7.0,
        denoise=1.0
    )

    # Queue and wait
    print(f"  Generating with IPAdapter...")
    prompt_id = client.queue_prompt(workflow)

    try:
        history = client.wait_for_completion(prompt_id, timeout=180)
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        return False

    # Get output image
    outputs = history.get('outputs', {})
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            # Download generated image
            for img_info in node_output['images']:
                filename = img_info['filename']
                subfolder = img_info.get('subfolder', '')

                img_data = client.get_image(filename, subfolder)

                # Save to output directory
                output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
                with open(output_path, 'wb') as f:
                    f.write(img_data)

                print(f"  ✓ Saved to {output_path}")
                return True

    print(f"  ✗ No output image found")
    return False


def main():
    """Generate all 25 clothed frames using IPAdapter."""
    # Check ComfyUI is running
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        print("ERROR: ComfyUI server not running at http://127.0.0.1:8188")
        print("Start it with: cd /Users/roberthyatt/Code/ComfyUI && python main.py")
        return 1

    frames_dir = Path("training_data/frames")
    masks_dir = Path("training_data/masks_inpainting")
    output_dir = Path("training_data/frames_ipadapter_generated")
    output_dir.mkdir(exist_ok=True)

    # Upload all 25 reference frames first
    print("=" * 70)
    print("UPLOADING REFERENCE FRAMES")
    print("=" * 70)
    print()

    for i in range(25):
        ref_path = frames_dir / f"clothed_frame_{i:02d}.png"
        if ref_path.exists():
            client.upload_image(ref_path)
            print(f"Uploaded reference frame {i:02d}")

    print()
    print("=" * 70)
    print("GENERATING CLOTHED FRAMES WITH IPADAPTER + CONTROLNET")
    print("=" * 70)
    print()

    success_count = 0

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        if generate_clothed_frame(client, frame_idx, frames_dir, masks_dir, output_dir):
            success_count += 1

        print()

    print("=" * 70)
    print(f"✓ Generated {success_count}/25 frames")
    print("=" * 70)

    return 0 if success_count == 25 else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Test manually (requires ComfyUI running)**

Run:
```bash
# First, generate masks
python generate_inpainting_masks.py

# Then generate one frame as test
python generate_with_ipadapter.py
```

Expected: Generates frame in `training_data/frames_ipadapter_generated/`

**Step 3: Commit**

```bash
git add generate_with_ipadapter.py
git commit -m "feat: add IPAdapter generation script

Generates all 25 clothed frames using:
- IPAdapter trained on reference frames
- OpenPose ControlNet for pose matching
- Inpainting for masked generation

Each frame generated with base skeleton pose."
```

---

### Task 4: Update Pipeline to Use IPAdapter Generation

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/process_clothing_spritesheet.py`

**Step 1: Replace skeleton warping with IPAdapter generation**

Modify Step 1 in `process_clothing_spritesheet.py`:

```python
# OLD:
from skeleton_warp_alignment import main as align_main

# NEW:
from generate_with_ipadapter import main as ipadapter_main
from generate_inpainting_masks import main as mask_gen_main
```

Update Step 1 section:

```python
        # Step 0.5: Generate inpainting masks
        print("\n" + "=" * 70)
        print("STEP 0.5/5: Generating inpainting masks")
        print("=" * 70 + "\n")

        result = mask_gen_main()
        if result != 0:
            print("ERROR: Mask generation failed")
            return 1

        # Step 1: Generate clothed frames with IPAdapter
        print("\n" + "=" * 70)
        print("STEP 1/5: Generating with IPAdapter + ControlNet")
        print("=" * 70 + "\n")

        result = ipadapter_main()
        if result != 0:
            print("ERROR: IPAdapter generation failed")
            return 1
```

Update Step 2 references:

```python
        # Step 2: Extend armor to cover feet
        print("\n" + "=" * 70)
        print("STEP 2/5: Extending armor to cover feet")
        print("=" * 70 + "\n")

        # Update extend_armor_feet.py to use frames_ipadapter_generated
        # (Will do in next task)
```

**Step 2: Update extend_armor_feet.py to use new directory**

Modify `extend_armor_feet.py`:

```python
def main():
    """Extend armor in all IPAdapter-generated frames."""
    frames_dir = Path("training_data/frames")
    generated_dir = Path("training_data/frames_ipadapter_generated")  # NEW
    output_dir = Path("training_data/frames_complete_ipadapter")  # NEW
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXTENDING ARMOR TO COVER FEET")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        generated_path = generated_dir / f"clothed_frame_{frame_idx:02d}.png"  # NEW

        base = Image.open(base_path)
        generated = Image.open(generated_path)  # NEW

        # Extend armor
        extended = extend_armor_to_cover_feet(generated, base)  # NEW

        # Rest unchanged...
```

**Step 3: Update pipeline Step 3 references**

```python
        # Step 3: Generate masks
        result = subprocess.run([
            sys.executable,
            "predict_masks_with_model.py",
            "--frames-dir", "training_data/frames_complete_ipadapter",  # NEW
            "--output-dir", "training_data_validation/masks_corrected"
        ])
```

**Step 4: Update pipeline Step 4 references**

```python
        # Step 4: Copy frames for validation
        for i in range(25):
            src = Path(f"training_data/frames_complete_ipadapter/clothed_frame_{i:02d}.png")  # NEW
            dst = val_frames / f"clothed_frame_{i:02d}.png"
            shutil.copy(src, dst)
```

**Step 5: Update pipeline Step 5 references**

```python
        # Step 5: Extract clothing
        output_path = Path("training_data/clothing_spritesheet_ipadapter.png")  # NEW
        extract_with_validated_masks(
            frames_dir=Path("training_data/frames_complete_ipadapter"),  # NEW
            masks_dir=Path("training_data_validation/masks_corrected"),
            output_path=output_path
        )
```

**Step 6: Commit**

```bash
git add process_clothing_spritesheet.py extend_armor_feet.py
git commit -m "refactor: update pipeline to use IPAdapter generation

Replaces skeleton warping with IPAdapter + ControlNet:
- Generates masks from alpha channel
- Generates clothed frames in base poses
- Rest of pipeline (extend, mask, extract) unchanged"
```

---

### Task 5: Test IPAdapter Generation Pipeline

**Files:**
- Test: Manual testing of full pipeline

**Step 1: Ensure ComfyUI is running with required models**

Verify:
```bash
# Check ComfyUI models directory
ls -la ComfyUI/models/ipadapter/
ls -la ComfyUI/models/controlnet/

# Required files:
# - ip-adapter_sd15.bin (or similar)
# - control_v11p_sd15_openpose.pth (or similar)
```

If missing, install:
```bash
# IPAdapter: Download from https://huggingface.co/h94/IP-Adapter
# ControlNet: Download from https://huggingface.co/lllyasviel/ControlNet-v1-1
```

**Step 2: Run pipeline on test frame (frame 0 only)**

Modify scripts to process only frame 0 for testing:

In `generate_with_ipadapter.py`:
```python
# Temporarily change:
for frame_idx in range(1):  # Was range(25)
```

In `extend_armor_feet.py`:
```python
# Temporarily change:
for frame_idx in range(1):  # Was range(25)
```

Run pipeline:
```bash
python process_clothing_spritesheet.py
```

**Step 3: Inspect generated frame**

Check output:
```bash
# View generated frame
open training_data/frames_ipadapter_generated/clothed_frame_00.png

# View extended frame
open training_data/frames_complete_ipadapter/clothed_frame_00.png
```

Verify:
- [ ] Generated frame has armor in correct pose
- [ ] Pixel art quality preserved (no blur)
- [ ] Armor style matches reference frames
- [ ] Extended frame covers feet properly

**Step 4: If quality is good, restore full processing**

Restore `range(25)` in both scripts.

**Step 5: If quality is poor, adjust parameters**

Common issues and fixes:

**Issue: Blurry output**
- Increase steps to 40-50
- Try sampler "euler_a" instead of "dpmpp_2m"
- Add "sharp, crisp, pixel art" to prompt

**Issue: Armor on head/hands/feet**
- Switch to Option A (precise masks)
- Modify `generate_inpainting_masks.py` to use body-part masks

**Issue: Doesn't match armor style**
- Increase IPAdapter weight to 0.9
- Add more descriptive prompt details
- Verify all 25 reference frames uploaded

**Step 6: Document findings**

Create `docs/findings/2025-12-09-ipadapter-generation-quality.md`:

```markdown
# IPAdapter Generation Quality Test

## Test Date
2025-12-09

## Parameters Tested
- Steps: 35
- CFG: 7.0
- Sampler: dpmpp_2m karras
- IPAdapter weight: 0.8
- ControlNet strength: 0.9

## Results
[Document what worked/didn't work]

## Adjustments Made
[List any parameter changes]

## Final Configuration
[Final working parameters]
```

---

### Task 6: Run Full Pipeline and Generate Spritesheet

**Files:**
- Execute: Complete pipeline run

**Step 1: Restore full frame processing**

Verify both scripts process all 25 frames:
```python
# In generate_with_ipadapter.py
for frame_idx in range(25):

# In extend_armor_feet.py
for frame_idx in range(25):
```

**Step 2: Run complete pipeline**

```bash
python process_clothing_spritesheet.py
```

This will:
1. Start ComfyUI server
2. Generate inpainting masks (25 frames)
3. Generate clothed frames with IPAdapter (25 frames)
4. Extend armor to cover feet (25 frames)
5. Generate masks with U-Net
6. Open validation tool (manual review)
7. Extract clothing layers
8. Create final spritesheet
9. Stop ComfyUI server

**Step 3: Manual validation**

When mask validation tool opens:
- Review each frame
- Correct any mask errors
- Save to proceed

**Step 4: Verify final spritesheet**

Check output:
```bash
open training_data/clothing_spritesheet_ipadapter.png
```

Verify:
- [ ] 5x5 grid with 25 frames
- [ ] Clothing matches all base poses
- [ ] No body parts peeking through
- [ ] Pixel art quality preserved
- [ ] Armor style consistent across frames

**Step 5: Create verification overlay**

```bash
# Should already be created by pipeline
open training_data/final_verification.png
```

Verify clothing composites correctly over base spritesheet.

**Step 6: Commit final outputs**

```bash
git add training_data/clothing_spritesheet_ipadapter.png
git add training_data/final_verification.png
git commit -m "feat: generate clothing spritesheet with IPAdapter

Complete pipeline run:
- IPAdapter + ControlNet generation in base poses
- U-Net masking for extraction
- 5x5 spritesheet output

Quality: [describe results]"
```

---

## Verification Checklist

After completing all tasks:

- [ ] Inpainting masks generated from alpha channel
- [ ] IPAdapter workflow includes all required nodes
- [ ] All 25 frames generated with correct poses
- [ ] Pixel art quality preserved (no blur/distortion)
- [ ] Armor style matches reference frames
- [ ] No body parts peeking through clothing
- [ ] U-Net masking extracts clothing cleanly
- [ ] Final spritesheet is 5x5 grid (2560x2560)
- [ ] Verification overlay composites correctly
- [ ] Pipeline runs end-to-end without errors

---

## Troubleshooting

### ComfyUI Models Missing

If IPAdapter or ControlNet models not found:

```bash
# Download IPAdapter
cd ComfyUI/models/ipadapter
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin

# Download ControlNet OpenPose
cd ComfyUI/models/controlnet
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
```

### Generation Quality Issues

See Task 5, Step 5 for parameter tuning guidance.

### Fallback to Option A (Precise Masks)

If full-body masking generates armor on head/hands:

1. Modify `generate_inpainting_masks.py`
2. Load your character body-part masks
3. Use only torso/limbs, exclude head/hands/feet

---

## Next Steps

After successful spritesheet generation:

1. Compare IPAdapter results with previous approaches
2. If quality is good, update main workflow to use IPAdapter
3. If quality needs work, iterate on parameters or try Option C (per-frame references)
4. Document final pipeline configuration
