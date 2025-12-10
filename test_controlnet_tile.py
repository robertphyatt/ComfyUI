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
