"""Test IPAdapter + ControlNet together (no mask)."""
import sys
import json
from pathlib import Path
from sprite_clothing_gen.comfy_client import ComfyUIClient

def build_test_workflow(reference_image: str, base_image: str, seed: int = 42):
    """IPAdapter + ControlNet, no mask."""
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
        
        # Load base image (for OpenPose)
        "4": {
            "inputs": {"image": base_image},
            "class_type": "LoadImage"
        },
        
        # ControlNet Loader
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
        
        # Apply IPAdapter FIRST
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
        
        # Apply ControlNet to conditioning
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
        
        # Empty latent (txt2img)
        "11": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        
        # KSampler - uses IPAdapter-modified model + ControlNet conditioning
        "12": {
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["7", 0],  # IPAdapter-modified model
                "positive": ["10", 0],  # ControlNet conditioning
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
                "filename_prefix": "ipadapter_test_with_controlnet",
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

print("Building test workflow (IPAdapter + ControlNet)...")
workflow = build_test_workflow("clothed_frame_00.png", "base_frame_00.png", seed=888)

print("Generating...")
prompt_id = client.queue_prompt(workflow)
history = client.wait_for_completion(prompt_id, timeout=120)

print("\n✓ Test complete - check output/ipadapter_test_with_controlnet_*.png")
print("If it has brown armor → ControlNet isn't the problem")
print("If it's naked/gray → ControlNet is blocking IPAdapter")
