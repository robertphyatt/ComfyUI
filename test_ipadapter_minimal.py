"""Minimal IPAdapter test - verify it works at all."""
import sys
import json
from pathlib import Path
from sprite_clothing_gen.comfy_client import ComfyUIClient

def build_minimal_test_workflow(reference_image: str, seed: int = 42):
    """Minimal workflow: IPAdapter + txt2img, nothing else."""
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
        
        # Load reference image
        "3": {
            "inputs": {"image": reference_image},
            "class_type": "LoadImage"
        },
        
        # Apply IPAdapter
        "4": {
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
        "5": {
            "inputs": {
                "text": "character wearing brown leather armor, pixel art",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        
        # Negative prompt
        "6": {
            "inputs": {
                "text": "blurry, low quality",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        
        # Empty latent (txt2img)
        "7": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        
        # KSampler
        "8": {
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],  # IPAdapter-modified model
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0]
            },
            "class_type": "KSampler"
        },
        
        # VAE Decode
        "9": {
            "inputs": {
                "samples": ["8", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },
        
        # Save
        "10": {
            "inputs": {
                "filename_prefix": "ipadapter_test_minimal",
                "images": ["9", 0]
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

print("Uploading reference image...")
client.upload_image(Path("input/clothed_frame_00.png"))

print("Building minimal test workflow...")
workflow = build_minimal_test_workflow("clothed_frame_00.png", seed=999)

print("Generating with IPAdapter ONLY (no ControlNet, no mask)...")
prompt_id = client.queue_prompt(workflow)
history = client.wait_for_completion(prompt_id, timeout=120)

print("\n✓ Test complete - check output/ipadapter_test_minimal_*.png")
print("If it has brown armor → IPAdapter works")
print("If it's random/unrelated → IPAdapter is broken")
