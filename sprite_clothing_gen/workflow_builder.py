"""Build ComfyUI workflows for clothing generation."""

from typing import Dict, Any
from sprite_clothing_gen.config import (
    CHECKPOINT_MODEL,
    CONTROLNET_MODEL,
    GENERATION_STEPS,
    CFG_SCALE,
    SAMPLER,
    SCHEDULER,
    DENOISE,
    CONTROLNET_STRENGTH,
    POSITIVE_PROMPT,
    NEGATIVE_PROMPT,
)


def build_clothing_generation_workflow(
    pose_image_filename: str,
    reference_image_filename: str,
    seed: int = 42,
    output_filename_prefix: str = "clothing"
) -> Dict[str, Any]:
    """Build ComfyUI workflow for generating clothing layer from pose.

    Workflow structure:
    1. Load checkpoint model
    2. Load ControlNet model
    3. Load pose image (OpenPose skeleton)
    4. Load reference clothing image
    5. Apply ControlNet conditioning
    6. Generate image with KSampler
    7. Save output

    Args:
        pose_image_filename: Filename of pose skeleton image (in ComfyUI input dir)
        reference_image_filename: Filename of reference clothing image (in ComfyUI input dir)
        seed: Random seed for generation
        output_filename_prefix: Prefix for output filename

    Returns:
        ComfyUI workflow dictionary
    """
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": CHECKPOINT_MODEL
            }
        },
        "2": {
            "class_type": "ControlNetLoader",
            "inputs": {
                "control_net_name": CONTROLNET_MODEL
            }
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {
                "image": pose_image_filename
            }
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {
                "image": reference_image_filename
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": POSITIVE_PROMPT,
                "clip": ["1", 1]
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": NEGATIVE_PROMPT,
                "clip": ["1", 1]
            }
        },
        "7": {
            "class_type": "ControlNetApply",
            "inputs": {
                "strength": CONTROLNET_STRENGTH,
                "conditioning": ["5", 0],
                "control_net": ["2", 0],
                "image": ["3", 0]
            }
        },
        "8": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["4", 0],
                "vae": ["1", 2]
            }
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": GENERATION_STEPS,
                "cfg": CFG_SCALE,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "denoise": DENOISE,
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["6", 0],
                "latent_image": ["8", 0]
            }
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["1", 2]
            }
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": output_filename_prefix,
                "images": ["10", 0]
            }
        }
    }

    return workflow


def build_openpose_preprocessing_workflow(
    input_image_filename: str,
    output_filename_prefix: str = "pose"
) -> Dict[str, Any]:
    """Build ComfyUI workflow for OpenPose preprocessing.

    Args:
        input_image_filename: Filename of input image (in ComfyUI input dir)
        output_filename_prefix: Prefix for output filename

    Returns:
        ComfyUI workflow dictionary
    """
    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": input_image_filename
            }
        },
        "2": {
            "class_type": "OpenposePreprocessor",
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "disable",
                "resolution": 512,
                "image": ["1", 0]
            }
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": output_filename_prefix,
                "images": ["2", 0]
            }
        }
    }

    return workflow


def build_ipadapter_generation_workflow(
    base_image_name: str,
    mask_image_name: str,
    reference_image_names: list[str],
    prompt: str = "Brown leather armor, pixel art",
    negative_prompt: str = "blurry, low quality",
    seed: int = 12345,
    steps: int = 35,
    cfg: float = 7.0,
    denoise: float = 1.0
) -> dict:
    """Build ComfyUI workflow for IPAdapter + ControlNet inpainting.

    Args:
        base_image_name: Filename of base character image
        mask_image_name: Filename of inpainting mask
        reference_image_names: List of 25 clothed reference frame filenames
        prompt: Positive text prompt
        negative_prompt: Negative text prompt
        seed: Random seed for reproducibility
        steps: Sampling steps
        cfg: CFG scale
        denoise: Denoise strength (1.0 = full generation)

    Returns:
        ComfyUI workflow dict
    """
    workflow = {
        # 1. Load base image
        "1": {
            "inputs": {"image": base_image_name},
            "class_type": "LoadImage"
        },

        # 2. Load inpainting mask
        "2": {
            "inputs": {"image": mask_image_name},
            "class_type": "LoadImage"
        },

        # 3. Load IPAdapter model
        "3": {
            "inputs": {"ipadapter_file": "ip-adapter_sd15.bin"},
            "class_type": "IPAdapterModelLoader"
        },
    }

    # 4-28: Load 25 reference images individually (clothed frames)
    # NOTE: Using Approach B from plan - LoadImageBatch doesn't support explicit filename lists
    # Instead, we create individual LoadImage nodes and batch them with ImageBatch
    # This properly uses the reference_image_names parameter instead of ignoring it

    # Start with node ID 4, allocate enough IDs for loaders and batchers
    # We need 25 LoadImage nodes + 24 ImageBatch nodes = 49 nodes total
    # Use nodes 4-28 for the 25 reference loaders
    reference_loader_ids = []
    for idx, ref_image_name in enumerate(reference_image_names):
        node_id = str(4 + idx)  # Nodes 4-28 for 25 reference images
        workflow[node_id] = {
            "inputs": {
                "image": ref_image_name
            },
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Reference Image {idx}"}
        }
        reference_loader_ids.append(node_id)

    # Build a binary tree of ImageBatch nodes to combine all 25 images
    # Use nodes starting from 29 and up for the batch operations
    # But wait - we need node 29 for checkpoint loader per the existing code
    # So we can't use 29+ for batching. Let me use a different range.

    # Actually, let's use a completely different strategy to avoid conflicts:
    # We'll allocate nodes in reverse order from where we need them
    # The IPAdapter Apply is at node 30, so the final batch result needs to be < 30

    # Let's use this layout:
    # Nodes 4-28: Individual LoadImage nodes (25 nodes)
    # We can't fit 24 batch nodes before node 29 as well
    # Solution: Make the final batch node be at a known location, say node 28
    # And work backwards

    # Better approach: rebuild the entire node ID scheme
    # Let's assign sequential IDs starting from 4:
    # 4-28: LoadImage for each reference (25 nodes)
    # After that, we batch them progressively

    # Actually, simplest solution: just insert all the batching nodes AFTER we define
    # them, and track what the final batch node ID will be, ensuring it's < 30

    # Let's use a simple approach: create the batch tree and track the final node
    current_batch_nodes = reference_loader_ids[:]
    next_node_id = 29  # Start batch nodes at 29... but that conflicts!

    # OK, new strategy: renumber the checkpoint and subsequent nodes
    # to come AFTER all our batching logic. But that changes the whole workflow.

    # Simpler fix: use negative lookbehind - build batches using high node IDs,
    # then store final result in a LoadImage node... no that doesn't work.

    # Let me reconsider: The constraint is that node 30 needs to reference the
    # final batch. If the final batch is node 223, that violates ordering.
    # Solution: move nodes 29+ to higher IDs, or renumber everything.

    # Cleanest solution: renumber the main workflow nodes to start at 500+
    # This gives us plenty of room for the batching tree.

    # Build batching tree using nodes starting at 29
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

    # Now add the rest of the workflow nodes, starting from batch_node_id
    # This ensures all node IDs are sequential and no circular dependencies
    checkpoint_node_id = str(batch_node_id)
    batch_node_id += 1
    ipadapter_apply_id = str(batch_node_id)
    batch_node_id += 1
    controlnet_loader_id = str(batch_node_id)
    batch_node_id += 1
    openpose_id = str(batch_node_id)
    batch_node_id += 1
    clip_positive_id = str(batch_node_id)
    batch_node_id += 1
    clip_negative_id = str(batch_node_id)
    batch_node_id += 1
    controlnet_apply_id = str(batch_node_id)
    batch_node_id += 1
    vae_encode_id = str(batch_node_id)
    batch_node_id += 1
    set_mask_id = str(batch_node_id)
    batch_node_id += 1
    ksampler_id = str(batch_node_id)
    batch_node_id += 1
    vae_decode_id = str(batch_node_id)
    batch_node_id += 1
    save_image_id = str(batch_node_id)

    # Continue building the rest of the workflow with dynamic node IDs
    workflow[checkpoint_node_id] = {
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"}
    }

    workflow[ipadapter_apply_id] = {
        "inputs": {
            "weight": 0.8,
            "weight_type": "style transfer",  # Fixed: use valid IPAdapter weight_type
            "start_at": 0.0,
            "end_at": 1.0,
            "ipadapter": ["3", 0],
            "image": [final_batch_node, 0],  # Reference images from final batch
            "model": [checkpoint_node_id, 0]
        },
        "class_type": "IPAdapter",  # Fixed: use correct node name
        "_meta": {"title": "IPAdapter"}
    }

    workflow[controlnet_loader_id] = {
        "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"},
        "class_type": "ControlNetLoader",
        "_meta": {"title": "Load ControlNet"}
    }

    workflow[openpose_id] = {
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "image": ["1", 0]  # Base image
        },
        "class_type": "OpenposePreprocessor",
        "_meta": {"title": "OpenPose Preprocessor"}
    }

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

    workflow[controlnet_apply_id] = {
        "inputs": {
            "strength": 0.9,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "positive": [clip_positive_id, 0],
            "negative": [clip_negative_id, 0],
            "control_net": [controlnet_loader_id, 0],
            "image": [openpose_id, 0]
        },
        "class_type": "ControlNetApplyAdvanced",
        "_meta": {"title": "Apply ControlNet"}
    }

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
            "latent_image": [set_mask_id, 0]
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler (Inpainting)"}
    }

    workflow[vae_decode_id] = {
        "inputs": {
            "samples": [ksampler_id, 0],
            "vae": [checkpoint_node_id, 2]
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"}
    }

    workflow[save_image_id] = {
        "inputs": {
            "filename_prefix": "ipadapter_generated",
            "images": [vae_decode_id, 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
    }

    return workflow
