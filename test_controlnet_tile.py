#!/usr/bin/env python3
"""Test ControlNet Tile vs OpenPose for pose accuracy."""

import sys
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

# Add ComfyUI to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sprite_clothing_gen.comfy_client import ComfyUIClient


def setup_logging():
    """Configure logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/tmp/tile_test_{timestamp}.log"

    class Logger:
        def __init__(self, log_file):
            self.log_file = log_file
            self.terminal = sys.stdout

        def write(self, message):
            self.terminal.write(message)
            with open(self.log_file, 'a') as f:
                f.write(message)

        def flush(self):
            self.terminal.flush()

    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Logging to {log_file}")
    return log_file


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def setup_test_directories():
    """Create clean test output directories."""
    base_dir = Path("output/tile_test")

    # Remove old test runs
    if base_dir.exists():
        log(f"Removing old test directory: {base_dir}")
        shutil.rmtree(base_dir)

    # Create subdirectories for each strength level
    strengths = [0.6, 0.8, 1.0]
    for strength in strengths:
        dir_path = base_dir / f"frame_04_strength_{strength}"
        dir_path.mkdir(parents=True, exist_ok=True)
        log(f"Created test directory: {dir_path}")

    return base_dir, strengths


def build_tile_workflow(
    base_image_name: str,
    reference_image_names: list[str],
    tile_strength: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int = 35,
    cfg: float = 7.0,
    denoise: float = 1.0
) -> dict:
    """Build ComfyUI workflow for IPAdapter + ControlNet Tile.

    Based on build_ipadapter_generation_workflow() but using Tile instead of OpenPose.
    """
    num_references = len(reference_image_names)
    max_batch_nodes = num_references - 1
    checkpoint_node_id = str(29 + max_batch_nodes)

    workflow = {
        # 1. Load base image
        "1": {
            "inputs": {"image": base_image_name},
            "class_type": "LoadImage"
        },

        # 2. Unused mask slot (kept for consistency with production workflow)
        "2": {
            "inputs": {"image": base_image_name},
            "class_type": "LoadImage"
        },

        # 3. Load IPAdapter with Unified Loader
        "3": {
            "inputs": {
                "model": [checkpoint_node_id, 0],
                "preset": "PLUS (high strength)"
            },
            "class_type": "IPAdapterUnifiedLoader",
            "_meta": {"title": "IPAdapter Unified Loader"}
        },
    }

    # 4-28: Load 25 reference images
    reference_loader_ids = []
    for idx, ref_image_name in enumerate(reference_image_names):
        node_id = str(4 + idx)
        workflow[node_id] = {
            "inputs": {"image": ref_image_name},
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Reference Image {idx}"}
        }
        reference_loader_ids.append(node_id)

    # Build batching tree
    current_batch_nodes = reference_loader_ids[:]
    batch_node_id = 29
    while len(current_batch_nodes) > 1:
        next_batch_nodes = []
        for i in range(0, len(current_batch_nodes), 2):
            if i + 1 < len(current_batch_nodes):
                workflow[str(batch_node_id)] = {
                    "inputs": {
                        "image1": [current_batch_nodes[i], 0],
                        "image2": [current_batch_nodes[i + 1], 0]
                    },
                    "class_type": "ImageBatch",
                    "_meta": {"title": f"Batch Step {batch_node_id}"}
                }
                next_batch_nodes.append(str(batch_node_id))
                batch_node_id += 1
            else:
                next_batch_nodes.append(current_batch_nodes[i])
        current_batch_nodes = next_batch_nodes

    final_batch_node = current_batch_nodes[0]

    # Assign remaining node IDs
    checkpoint_node_id = str(batch_node_id)
    batch_node_id += 1
    ipadapter_apply_id = str(batch_node_id)
    batch_node_id += 1
    controlnet_loader_id = str(batch_node_id)
    batch_node_id += 1
    tile_preprocessor_id = str(batch_node_id)
    batch_node_id += 1
    clip_positive_id = str(batch_node_id)
    batch_node_id += 1
    clip_negative_id = str(batch_node_id)
    batch_node_id += 1
    controlnet_apply_id = str(batch_node_id)
    batch_node_id += 1
    empty_latent_id = str(batch_node_id)
    batch_node_id += 1
    ksampler_id = str(batch_node_id)
    batch_node_id += 1
    vae_decode_id = str(batch_node_id)
    batch_node_id += 1
    save_image_id = str(batch_node_id)
    batch_node_id += 1
    # Debug SaveImage node IDs
    save_refs_id = str(batch_node_id)
    batch_node_id += 1
    save_tile_id = str(batch_node_id)
    batch_node_id += 1
    save_base_id = str(batch_node_id)

    # Checkpoint
    workflow[checkpoint_node_id] = {
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"}
    }

    # IPAdapter Apply
    workflow[ipadapter_apply_id] = {
        "inputs": {
            "weight": 1.0,
            "weight_type": "style and composition",
            "combine_embeds": "concat",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "V only",
            "ipadapter": ["3", 1],
            "image": [final_batch_node, 0],
            "model": ["3", 0]
        },
        "class_type": "IPAdapterAdvanced",
        "_meta": {"title": "IPAdapter Advanced"}
    }

    # Save reference batch (debug)
    workflow[save_refs_id] = {
        "inputs": {
            "filename_prefix": "tile_test/reference_batch",
            "images": [final_batch_node, 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Reference Batch (Debug)"}
    }

    # ControlNet Tile Loader
    workflow[controlnet_loader_id] = {
        "inputs": {"control_net_name": "control_v11f1e_sd15_tile.pth"},
        "class_type": "ControlNetLoader",
        "_meta": {"title": "Load ControlNet Tile"}
    }

    # Tile Preprocessor
    workflow[tile_preprocessor_id] = {
        "inputs": {
            "pyrUp_iters": 3,
            "resolution": 512,
            "image": ["1", 0]
        },
        "class_type": "TilePreprocessor",
        "_meta": {"title": "Tile Preprocessor"}
    }

    # Save Tile preprocessor output (debug)
    workflow[save_tile_id] = {
        "inputs": {
            "filename_prefix": "tile_test/tile_downsampled",
            "images": [tile_preprocessor_id, 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Tile Preprocessor (Debug)"}
    }

    # CLIP Text Encode
    workflow[clip_positive_id] = {
        "inputs": {
            "text": prompt,
            "clip": [checkpoint_node_id, 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive)"}
    }

    workflow[clip_negative_id] = {
        "inputs": {
            "text": negative_prompt,
            "clip": [checkpoint_node_id, 1]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative)"}
    }

    # ControlNet Apply
    workflow[controlnet_apply_id] = {
        "inputs": {
            "strength": tile_strength,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "positive": [clip_positive_id, 0],
            "negative": [clip_negative_id, 0],
            "control_net": [controlnet_loader_id, 0],
            "image": [tile_preprocessor_id, 0]
        },
        "class_type": "ControlNetApplyAdvanced",
        "_meta": {"title": f"Apply ControlNet Tile (strength={tile_strength})"}
    }

    # Empty Latent
    workflow[empty_latent_id] = {
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent Image"}
    }

    # KSampler
    workflow[ksampler_id] = {
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": denoise,
            "model": [ipadapter_apply_id, 0],
            "positive": [controlnet_apply_id, 0],
            "negative": [controlnet_apply_id, 1],
            "latent_image": [empty_latent_id, 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler (txt2img)"}
    }

    # VAE Decode
    workflow[vae_decode_id] = {
        "inputs": {
            "samples": [ksampler_id, 0],
            "vae": [checkpoint_node_id, 2]
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"}
    }

    # Save base input (debug)
    workflow[save_base_id] = {
        "inputs": {
            "filename_prefix": "tile_test/base_input",
            "images": ["1", 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Base Input (Debug)"}
    }

    # Save final output
    workflow[save_image_id] = {
        "inputs": {
            "filename_prefix": "tile_test/clothed_frame",
            "images": [vae_decode_id, 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
    }

    return workflow


def run_tile_test(strength: float, test_dir: Path, client: ComfyUIClient):
    """Run single test with given Tile strength."""
    log(f"\n{'='*70}")
    log(f"TESTING TILE STRENGTH: {strength}")
    log(f"{'='*70}")

    # Test parameters matching production
    frame_idx = 4
    base_name = f"base_frame_{frame_idx:02d}.png"
    reference_names = [f"clothed_frame_{i:02d}.png" for i in range(25)]

    prompt = "character wearing brown leather armor, pixel art"
    negative_prompt = "blurry, low quality, distorted, deformed, multiple heads, extra limbs, modern clothing, smooth, rendered, 3d, photorealistic"
    seed = 42 + frame_idx

    output_dir = test_dir / f"frame_{frame_idx:02d}_strength_{strength}"

    log(f"Base image: {base_name}")
    log(f"Reference images: {len(reference_names)} frames")
    log(f"Prompt: {prompt}")
    log(f"Seed: {seed}")
    log(f"Output directory: {output_dir}")

    # Upload base frame
    log("Uploading base frame...")
    base_path = Path("training_data/frames") / base_name
    if not base_path.exists():
        log(f"ERROR: Base frame not found: {base_path}")
        return False

    base_uploaded_name = client.upload_image(base_path)
    log(f"  Uploaded as: {base_uploaded_name}")

    # Upload reference frames
    log("Uploading reference frames...")
    ref_uploaded_names = []
    for ref_name in reference_names:
        ref_path = Path("training_data/frames") / ref_name
        if not ref_path.exists():
            log(f"ERROR: Reference frame not found: {ref_path}")
            return False
        ref_uploaded_name = client.upload_image(ref_path)
        ref_uploaded_names.append(ref_uploaded_name)
    log(f"  Uploaded {len(ref_uploaded_names)} reference frames")

    # Build workflow
    log("Building Tile workflow...")
    workflow = build_tile_workflow(
        base_image_name=base_uploaded_name,
        reference_image_names=ref_uploaded_names,
        tile_strength=strength,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=35,
        cfg=7.0,
        denoise=1.0
    )

    # Save workflow JSON
    workflow_path = output_dir / "workflow.json"
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    log(f"  Workflow saved: {workflow_path}")
    log(f"  Total nodes: {len(workflow)}")

    # Queue prompt
    log("Submitting to ComfyUI...")
    prompt_id = client.queue_prompt(workflow)
    log(f"  Prompt ID: {prompt_id}")

    # Wait for completion
    log("Waiting for generation to complete...")
    try:
        history = client.wait_for_completion(prompt_id, timeout=300)
        outputs = history.get("outputs", {})
        log(f"  Execution completed with {len(outputs)} output nodes")

        # Move artifacts to test directory
        log("Moving artifacts to test directory...")
        comfy_output = Path("output")

        # Find all tile_test artifacts from this run
        artifact_count = 0
        for artifact_path in comfy_output.glob("tile_test/*"):
            if artifact_path.is_file():
                dest_path = output_dir / artifact_path.name
                shutil.copy2(artifact_path, dest_path)
                log(f"  Saved: {dest_path.name} ({dest_path.stat().st_size} bytes)")
                artifact_count += 1

        if artifact_count == 0:
            log("  WARNING: No artifacts found in output/tile_test/")

        log(f"✓ Test complete for strength {strength}")
        return True

    except Exception as e:
        log(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    log_file = setup_logging()
    log("ControlNet Tile Test Starting")
    log(f"Working directory: {os.getcwd()}")

    # Initialize ComfyUI client
    log("Initializing ComfyUI client...")
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        log("ERROR: ComfyUI server is not running at http://127.0.0.1:8188")
        log("Please start ComfyUI and try again")
        sys.exit(1)
    log("  ComfyUI server is running")

    test_dir, strengths = setup_test_directories()
    log(f"Test output directory: {test_dir}")
    log(f"Testing strengths: {strengths}")

    # Run tests
    results = {}
    for strength in strengths:
        success = run_tile_test(strength, test_dir, client)
        results[strength] = success

    # Summary
    log(f"\n{'='*70}")
    log("TEST SUMMARY")
    log(f"{'='*70}")
    for strength, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        log(f"  Strength {strength}: {status}")

    log(f"\nResults saved to: {test_dir}")
    log(f"Log file: {log_file}")
    log("\nControlNet Tile Test Complete")
