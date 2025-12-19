#!/usr/bin/env python3
"""Test Nova Pixels XL pixelization via ComfyUI API."""

import json
import requests
import time
import uuid
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64
import websocket


COMFYUI_URL = "http://127.0.0.1:8188"


def upload_image(image_path: Path, name: str = None) -> str:
    """Upload image to ComfyUI and return filename."""
    if name is None:
        name = image_path.name

    with open(image_path, 'rb') as f:
        files = {'image': (name, f, 'image/png')}
        data = {'overwrite': 'true'}
        response = requests.post(f"{COMFYUI_URL}/upload/image", files=files, data=data)

    if response.status_code == 200:
        return response.json()['name']
    else:
        raise Exception(f"Upload failed: {response.text}")


def create_workflow(input_image: str, output_prefix: str, denoise: float = 0.65):
    """Create Nova Pixels XL img2img workflow."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "novaPixelsXL_v30.safetensors"
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "pixel art, pixelated, sprite, 8-bit"
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "blurry, smooth, anti-aliased, photorealistic, 3d render, anime, illustration, watermark, signature, text"
            }
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {
                "image": input_image
            }
        },
        "5": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["4", 0],
                "vae": ["1", 2]
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0],
                "seed": 42,
                "steps": 25,
                "cfg": 5,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": denoise
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2]
            }
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": output_prefix
            }
        }
    }


def queue_prompt(workflow: dict) -> str:
    """Queue a prompt and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    response = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
    if response.status_code == 200:
        return response.json()['prompt_id'], client_id
    else:
        raise Exception(f"Queue failed: {response.text}")


def wait_for_completion(prompt_id: str, client_id: str, timeout: int = 300):
    """Wait for prompt to complete using websocket."""
    ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url)

    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            result = ws.recv()
            if isinstance(result, str):
                msg = json.loads(result)
                if msg.get('type') == 'executing':
                    data = msg.get('data', {})
                    if data.get('prompt_id') == prompt_id and data.get('node') is None:
                        # Execution complete
                        return True
            time.sleep(0.1)
    finally:
        ws.close()

    return False


def get_output_images(prompt_id: str) -> list:
    """Get output images from completed prompt."""
    response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    if response.status_code != 200:
        return []

    history = response.json()
    if prompt_id not in history:
        return []

    outputs = history[prompt_id].get('outputs', {})
    images = []

    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            for img_info in node_output['images']:
                filename = img_info['filename']
                subfolder = img_info.get('subfolder', '')
                img_type = img_info.get('type', 'output')

                # Download image
                params = {'filename': filename, 'subfolder': subfolder, 'type': img_type}
                response = requests.get(f"{COMFYUI_URL}/view", params=params)
                if response.status_code == 200:
                    images.append(response.content)

    return images


def run_nova_pixels(input_path: Path, output_prefix: str, denoise: float = 0.65) -> bytes:
    """Run Nova Pixels XL on an image and return result."""
    print(f"  Uploading {input_path.name}...")
    uploaded_name = upload_image(input_path)

    print(f"  Creating workflow (denoise={denoise})...")
    workflow = create_workflow(uploaded_name, output_prefix, denoise)

    print(f"  Queuing prompt...")
    prompt_id, client_id = queue_prompt(workflow)

    print(f"  Waiting for completion...")
    if not wait_for_completion(prompt_id, client_id):
        raise Exception("Timeout waiting for completion")

    print(f"  Getting output images...")
    images = get_output_images(prompt_id)

    if not images:
        raise Exception("No output images")

    return images[0]


def main():
    base_dir = Path(__file__).parent.parent / "training_data" / "skeleton_comparison"
    output_dir = base_dir

    # Test images to process
    test_images = [
        "clothed_frame_00_expand_texture_only.png",
        "clothed_frame_01_expand_texture_only.png",
    ]

    # Also need the originals for comparison - let me use the no_warp versions
    # Actually let's process the texture_borrow results
    test_images = []
    for frame in ["clothed_frame_00", "clothed_frame_01"]:
        # Use the texture_borrow_only output
        path = base_dir / f"{frame}_texture_borrow_only.png"
        if path.exists():
            test_images.append(path)
        else:
            # Fall back to shift_inpaint
            path = base_dir / f"{frame}_shift_inpaint_only.png"
            if path.exists():
                test_images.append(path)

    if not test_images:
        print("No test images found!")
        return

    results = []

    # Test different denoise levels - try very low for preservation
    denoise_levels = [0.10, 0.15, 0.20]

    for img_path in test_images:
        for denoise in denoise_levels:
            print(f"\n=== Processing {img_path.name} (denoise={denoise}) ===")

            try:
                output_prefix = f"nova_{img_path.stem}_d{int(denoise*100)}"
                result_bytes = run_nova_pixels(img_path, output_prefix, denoise=denoise)

                # Save result
                output_path = output_dir / f"{img_path.stem}_nova_d{int(denoise*100)}.png"
                with open(output_path, 'wb') as f:
                    f.write(result_bytes)
                print(f"  Saved: {output_path.name}")

                results.append((img_path, output_path, denoise))

            except Exception as e:
                print(f"  ERROR: {e}")

    # Create side-by-side comparisons
    print("\n=== Creating comparisons ===")
    for orig_path, nova_path, denoise in results:
        orig = cv2.imread(str(orig_path), cv2.IMREAD_UNCHANGED)
        nova = cv2.imread(str(nova_path), cv2.IMREAD_UNCHANGED)

        if orig is None or nova is None:
            continue

        # Resize nova to match orig if needed (SDXL outputs 1024x1024)
        if nova.shape[:2] != orig.shape[:2]:
            nova = cv2.resize(nova, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure both have same number of channels
        if len(orig.shape) == 3 and orig.shape[2] == 4 and len(nova.shape) == 3 and nova.shape[2] == 3:
            # Add alpha channel to nova
            nova_alpha = np.ones((nova.shape[0], nova.shape[1], 1), dtype=np.uint8) * 255
            nova = np.concatenate([nova, nova_alpha], axis=2)
        elif len(nova.shape) == 3 and nova.shape[2] == 4 and len(orig.shape) == 3 and orig.shape[2] == 3:
            # Add alpha channel to orig
            orig_alpha = np.ones((orig.shape[0], orig.shape[1], 1), dtype=np.uint8) * 255
            orig = np.concatenate([orig, orig_alpha], axis=2)

        # Add labels
        def add_label(img, label):
            result = img.copy()
            cv2.putText(result, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1)
            return result

        orig_labeled = add_label(orig, "original")
        nova_labeled = add_label(nova, "nova_pixels")

        comparison = np.hstack([orig_labeled, nova_labeled])

        comp_path = output_dir / f"{orig_path.stem}_vs_nova.png"
        cv2.imwrite(str(comp_path), comparison)
        print(f"Saved comparison: {comp_path.name}")


if __name__ == "__main__":
    main()
