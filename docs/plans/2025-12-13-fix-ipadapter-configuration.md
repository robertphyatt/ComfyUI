# Fix IPAdapter Configuration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the IPAdapter configuration in workflow_builder.py to match the working test configuration, enabling proper armor style transfer.

**Architecture:** Update the IPAdapter node from basic `IPAdapter` to `IPAdapterAdvanced` with correct parameters, and change from img2img inpainting mode to txt2img generation mode.

**Tech Stack:** Python, ComfyUI API, IPAdapter Plus custom nodes

---

## Background

Investigation found that `workflow_builder.py` has wrong IPAdapter configuration compared to working `test_ipadapter_controlnet.py`:

| Setting | Working | Broken |
|---------|---------|--------|
| class_type | `IPAdapterAdvanced` | `IPAdapter` |
| weight | `1.0` | `0.8` |
| weight_type | `"style and composition"` | `"style transfer"` |
| combine_embeds | `"concat"` | MISSING |
| embeds_scaling | `"V only"` | MISSING |
| Latent source | `EmptyLatentImage` | `VAEEncode` + `SetLatentNoiseMask` |

---

### Task 1: Fix IPAdapter Node Configuration

**Files:**
- Modify: `sprite_clothing_gen/workflow_builder.py:348-360`

**Step 1: Update IPAdapter node to IPAdapterAdvanced with correct params**

Change lines 348-360 from:

```python
    workflow[ipadapter_apply_id] = {
        "inputs": {
            "weight": 0.8,
            "weight_type": "style transfer",  # Fixed: use valid IPAdapter weight_type
            "start_at": 0.0,
            "end_at": 1.0,
            "ipadapter": ["3", 1],  # IPAdapter dict from IPAdapterUnifiedLoader output [1]
            "image": [final_batch_node, 0],  # Reference images from final batch
            "model": ["3", 0]  # Model from IPAdapterUnifiedLoader output [0]
        },
        "class_type": "IPAdapter",  # Fixed: use correct node name
        "_meta": {"title": "IPAdapter"}
    }
```

To:

```python
    workflow[ipadapter_apply_id] = {
        "inputs": {
            "weight": 1.0,  # Full strength for clothing transfer
            "weight_type": "style and composition",  # Transfer both clothing style AND structure
            "combine_embeds": "concat",  # Properly combine multiple reference images
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",  # Standard scaling for batched images
            "ipadapter": ["3", 1],  # IPAdapter dict from IPAdapterUnifiedLoader output [1]
            "image": [final_batch_node, 0],  # Reference images from final batch
            "model": ["3", 0]  # Model from IPAdapterUnifiedLoader output [0]
        },
        "class_type": "IPAdapterAdvanced",  # Advanced node for multi-image style transfer
        "_meta": {"title": "IPAdapter Advanced"}
    }
```

**Step 2: Verify change with diagnostic script**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow
workflow = build_ipadapter_generation_workflow('base.png', 'mask.png', [f'ref_{i}.png' for i in range(25)])
for k, v in workflow.items():
    if 'IPAdapter' in v.get('class_type', ''):
        print(f'class_type: {v[\"class_type\"]}')
        print(f'weight: {v[\"inputs\"][\"weight\"]}')
        print(f'weight_type: {v[\"inputs\"][\"weight_type\"]}')
        print(f'combine_embeds: {v[\"inputs\"].get(\"combine_embeds\", \"MISSING\")}')
        print(f'embeds_scaling: {v[\"inputs\"].get(\"embeds_scaling\", \"MISSING\")}')
"
```

Expected output:
```
class_type: IPAdapterAdvanced
weight: 1.0
weight_type: style and composition
combine_embeds: concat
embeds_scaling: V only
```

---

### Task 2: Fix Workflow Mode (img2img to txt2img)

**Files:**
- Modify: `sprite_clothing_gen/workflow_builder.py:319-322` (node ID allocation)
- Modify: `sprite_clothing_gen/workflow_builder.py:414-432` (VAEEncode/SetLatentNoiseMask to EmptyLatentImage)

**Step 1: Update node ID allocation to remove set_mask_id**

Change lines 319-322 from:

```python
    vae_encode_id = str(batch_node_id)
    batch_node_id += 1
    set_mask_id = str(batch_node_id)
    batch_node_id += 1
```

To:

```python
    empty_latent_id = str(batch_node_id)  # txt2img mode - generate from empty latent
    batch_node_id += 1
    # set_mask_id removed - using txt2img instead of img2img inpainting
```

**Step 2: Replace VAEEncode + SetLatentNoiseMask with EmptyLatentImage**

Change lines 414-432 from:

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
            "mask": ["2", 1]  # Mask image (alpha channel)
        },
        "class_type": "SetLatentNoiseMask",
        "_meta": {"title": "Set Latent Noise Mask"}
    }
```

To:

```python
    # txt2img mode: generate from empty latent guided by pose, not inpainting
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
```

**Step 3: Update KSampler to use empty_latent_id**

Change the KSampler node (around line 434-448) latent_image input from:

```python
            "latent_image": [set_mask_id, 0]
```

To:

```python
            "latent_image": [empty_latent_id, 0]  # Generate from scratch, not inpaint
```

Also update the title from `"KSampler (Inpainting)"` to `"KSampler (txt2img)"`.

**Step 4: Verify workflow mode with diagnostic script**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow
workflow = build_ipadapter_generation_workflow('base.png', 'mask.png', [f'ref_{i}.png' for i in range(25)])
for k, v in workflow.items():
    ct = v.get('class_type', '')
    if ct in ['EmptyLatentImage', 'VAEEncode', 'SetLatentNoiseMask']:
        print(f'{ct}: present')
print('---')
for k, v in workflow.items():
    if v.get('class_type') == 'KSampler':
        latent_ref = v['inputs']['latent_image']
        latent_node = workflow[latent_ref[0]]
        print(f'KSampler latent source: {latent_node[\"class_type\"]}')
"
```

Expected output:
```
EmptyLatentImage: present
---
KSampler latent source: EmptyLatentImage
```

---

### Task 3: End-to-End Verification

**Step 1: Clear artifacts**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && rm -rf training_data/frames_ipadapter_generated output/ipadapter_generated_*.png
```

**Step 2: Run single frame test**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from pathlib import Path
from sprite_clothing_gen.comfy_client import ComfyUIClient
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow

client = ComfyUIClient('http://127.0.0.1:8188')
if not client.health_check():
    print('ERROR: ComfyUI not running')
    exit(1)

# Upload images
client.upload_image(Path('training_data/frames/base_frame_00.png'))
client.upload_image(Path('training_data/masks_inpainting/mask_00.png'))
for i in range(25):
    client.upload_image(Path(f'training_data/frames/clothed_frame_{i:02d}.png'))

# Build and run workflow
workflow = build_ipadapter_generation_workflow(
    'base_frame_00.png', 'mask_00.png',
    [f'clothed_frame_{i:02d}.png' for i in range(25)],
    prompt='character wearing brown leather armor, pixel art',
    negative_prompt='blurry, low quality',
    seed=42, steps=35, cfg=7.0, denoise=1.0
)

prompt_id = client.queue_prompt(workflow)
history = client.wait_for_completion(prompt_id, timeout=180)
print('Generation complete - check output/ipadapter_generated_*.png')
"
```

**Step 3: Verify output has armor**

Visually inspect `output/ipadapter_generated_00001_.png` - it should show brown leather armor, not naked gray mannequin.

---

### Task 4: Commit Changes

**Step 1: Commit the fix**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && git add sprite_clothing_gen/workflow_builder.py && git commit -m "fix: restore working IPAdapter configuration for armor transfer

- Change IPAdapter to IPAdapterAdvanced
- Set weight to 1.0 (was 0.8)
- Set weight_type to 'style and composition' (was 'style transfer')
- Add combine_embeds: 'concat' for multi-image batching
- Add embeds_scaling: 'V only'
- Switch from img2img (VAEEncode+SetLatentNoiseMask) to txt2img (EmptyLatentImage)

Root cause: These settings were in local uncommitted changes that got
overwritten during git checkout operations. The working configuration
was validated in test_ipadapter_controlnet.py."
```
