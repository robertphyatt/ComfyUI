"""Test img2img with denoise=0.65 vs txt2img."""
import sys
from pathlib import Path
from sprite_clothing_gen.comfy_client import ComfyUIClient

def build_img2img_workflow(reference_image: str, base_image: str, denoise: float, seed: int = 42):
    """IPAdapter + ControlNet OpenPose + img2img with configurable denoise."""
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

        # ControlNet Loader - OpenPose (winner from Task 1)
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
client.upload_image(Path("input/base_frame_00.png"))

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
print("\nBaseline comparison:")
print("  - txt2img baseline: output/ipadapter_test_with_controlnet_00001_.png")
