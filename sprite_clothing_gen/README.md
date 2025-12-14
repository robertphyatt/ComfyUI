# Sprite Clothing Generator

Automated pipeline to generate clothing-only sprite layers from a base spritesheet and single clothed reference frame, using OpenPose ControlNet for pose-aware generation.

## Overview

This tool takes:
- A 5x5 base spritesheet (25 frames) of a naked/base character
- A single reference frame with clothing applied (from Ludo or similar tool)
- The frame index that the reference corresponds to

And produces:
- A 5x5 clothing-only spritesheet with transparent backgrounds
- Clothing layers that perfectly align with each pose in the base spritesheet

You can then layer the clothing spritesheet over the base in Godot to create a fully clothed character.

## How It Works

1. **Split**: Base spritesheet split into 25 individual frames
2. **Extract**: U2-Net removes character body from reference, leaving only clothing
3. **Analyze**: OpenPose extracts pose skeletons from all 25 base frames
4. **Generate**: ControlNet generates clothing for each frame using:
   - Pose skeleton to match the exact pose
   - Clothing reference for style/color/design
   - Fixed seed for consistency across all frames
5. **Reassemble**: 25 clothing frames reassembled into 5x5 spritesheet

## Requirements

### ComfyUI Setup

1. **ComfyUI** running at http://127.0.0.1:8188
2. **Models installed**:
   - Checkpoint: `models/checkpoints/PixelartSpritesheet_V.1.ckpt`
   - ControlNet: `models/controlnet/control_v11p_sd15_openpose_fp16.safetensors`
3. **Custom nodes**:
   - `comfyui_controlnet_aux` (OpenPose preprocessor)
   - `rembg-comfyui-node-better` (U2-Net background removal)

### Python Dependencies

```bash
pip install rembg pillow requests
```

## Usage

### Command Line

```bash
python generate_sprite_clothing.py \
    --base input/base_spritesheet.png \
    --reference input/clothed_frame.png \
    --frame 12 \
    --output output/clothing_spritesheet.png \
    --seed 42
```

### Arguments

- `--base`: Path to base 5x5 spritesheet (25 frames, naked character)
- `--reference`: Path to single clothed reference frame from Ludo
- `--frame`: Frame index that reference corresponds to (0-24, zero-indexed)
- `--output`: Path to save output clothing spritesheet (default: output/clothing_spritesheet.png)
- `--seed`: Random seed for generation consistency (default: 42)
- `--keep-temp`: Keep temporary files for debugging (optional)
- `--comfyui-url`: ComfyUI server URL (default: http://127.0.0.1:8188)

### Python API

```python
from sprite_clothing_gen.orchestrator import SpriteClothingGenerator
from pathlib import Path

generator = SpriteClothingGenerator()

result = generator.generate(
    base_spritesheet=Path("input/base_spritesheet.png"),
    reference_frame=Path("input/clothed_frame.png"),
    reference_frame_index=12,
    output_path=Path("output/clothing_spritesheet.png"),
    seed=42
)
```

## Configuration

Edit `sprite_clothing_gen/config.py` to adjust:

- Generation parameters (steps, CFG scale, sampler)
- ControlNet strength
- Prompts (positive/negative)
- Model names
- Grid size (if not 5x5)

## Testing

```bash
# Run unit tests
pytest tests/sprite_clothing_gen/ -v

# Run integration tests (requires ComfyUI running)
pytest tests/test_integration.py -m integration -v
```

## Workflow Details

### OpenPose Preprocessing

Extracts body/hand poses from each frame. Produces skeleton images showing joint positions and limb connections.

### U2-Net Clothing Segmentation

Uses `u2net_cloth_seg` model to separate clothing from character body in the reference frame.

### ControlNet Generation

For each of the 25 frames:
- Conditions on OpenPose skeleton (exact pose matching)
- Uses clothing reference for style guidance
- Fixed seed ensures consistency
- Negative prompts suppress body parts

## Troubleshooting

### ComfyUI Connection Error

```
RuntimeError: ComfyUI server not accessible at http://127.0.0.1:8188
```

**Solution**: Start ComfyUI server before running the script

```bash
cd /Users/roberthyatt/Code/ComfyUI
python main.py
```

### Model Not Found

```
RuntimeError: Model not found: PixelartSpritesheet_V.1.ckpt
```

**Solution**: Download model and place in `models/checkpoints/`

### Clothing Includes Body Parts

If generated clothing layers include character body parts (head, hands):

1. Adjust `NEGATIVE_PROMPT` in `config.py` to be more explicit
2. Increase `CONTROLNET_STRENGTH` for stricter pose matching
3. Verify U2-Net properly segmented the reference frame (check temp files with `--keep-temp`)

### Inconsistent Style Across Frames

If clothing style varies between frames:

1. Use fixed `--seed` value (same seed = consistent results)
2. Verify reference frame is clear and high quality
3. Adjust `DENOISE` parameter in config.py (lower = more faithful to reference)

## Directory Structure

```
sprite_clothing_gen/
├── __init__.py
├── config.py              # Configuration settings
├── comfy_client.py        # ComfyUI API client
├── spritesheet_utils.py   # Split/reassemble utilities
├── clothing_extractor.py  # U2-Net segmentation
├── workflow_builder.py    # ComfyUI workflow construction
├── orchestrator.py        # Main pipeline orchestration
├── temp/                  # Temporary files (created at runtime)
│   ├── frames/           # Individual base frames
│   ├── poses/            # OpenPose skeletons
│   ├── clothing/         # Generated clothing layers
│   └── reference_clothing_only.png
└── README.md
```

## License

See main ComfyUI repository for license information.
