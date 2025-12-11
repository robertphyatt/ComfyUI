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
        "inputs": {"control_net_name": "control_v11f1e_sd15_tile.safetensors"},
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


if __name__ == "__main__":
    log_file = setup_logging()
    log("ControlNet Tile Test Starting")
    log(f"Working directory: {os.getcwd()}")

    test_dir, strengths = setup_test_directories()
    log(f"Test output directory: {test_dir}")
    log(f"Testing strengths: {strengths}")
