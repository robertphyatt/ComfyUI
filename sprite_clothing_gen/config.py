"""Configuration for sprite clothing generator."""

from pathlib import Path

# ComfyUI settings
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_API_URL = f"{COMFYUI_URL}/api"

# Model paths (relative to ComfyUI root)
CHECKPOINT_MODEL = "PixelartSpritesheet_V.1.ckpt"
CONTROLNET_MODEL = "control_v11p_sd15_openpose_fp16.safetensors"

# Spritesheet settings
GRID_SIZE = (5, 5)  # 5x5 grid = 25 frames
FRAME_COUNT = GRID_SIZE[0] * GRID_SIZE[1]

# Generation settings
GENERATION_STEPS = 25
CFG_SCALE = 7.5
SAMPLER = "euler_ancestral"
SCHEDULER = "normal"
DENOISE = 0.75
CONTROLNET_STRENGTH = 0.9

# Prompts
POSITIVE_PROMPT = "pixel art clothing layer, transparent background, clothing only, high quality pixel art"
NEGATIVE_PROMPT = "character, head, face, hands, skin, body, person, human, background, floor, ground"

# U2-Net model for clothing segmentation
U2NET_MODEL = "u2net_cloth_seg"

# Temporary file settings
TEMP_DIR = Path(__file__).parent / "temp"
KEEP_TEMP_FILES = False
