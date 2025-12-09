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

        # 4-28: Load 25 reference images (clothed frames)
        # For brevity, we'll load them in a batch node
        "4": {
            "inputs": {
                "mode": "incremental_image",
                "index": 0,
                "label": "batch",
                "path": "input/",
                "pattern": "clothed_frame_*.png",
                "allow_RGBA_output": "false"
            },
            "class_type": "LoadImageBatch"
        },

        # 29. Apply IPAdapter
        "29": {
            "inputs": {
                "weight": 0.8,
                "weight_type": "linear",
                "start_at": 0.0,
                "end_at": 1.0,
                "unfold_batch": "false",
                "ipadapter": ["3", 0],
                "image": ["4", 0],  # Reference images
                "model": ["30", 0]  # Will connect to checkpoint loader
            },
            "class_type": "IPAdapterApply"
        },

        # 30. Load checkpoint
        "30": {
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },

        # 31. Load ControlNet (OpenPose)
        "31": {
            "inputs": {"control_net_name": "control_v11p_sd15_openpose.pth"},
            "class_type": "ControlNetLoader"
        },

        # 32. OpenPose Preprocessor (extract skeleton from base)
        "32": {
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "image": ["1", 0]  # Base image
            },
            "class_type": "OpenposePreprocessor"
        },

        # 33. Apply ControlNet
        "33": {
            "inputs": {
                "strength": 0.9,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "positive": ["34", 0],  # Will connect to CLIP
                "negative": ["35", 0],
                "control_net": ["31", 0],
                "image": ["32", 0]  # OpenPose skeleton
            },
            "class_type": "ControlNetApplyAdvanced"
        },

        # 34. CLIP Text Encode (positive prompt)
        "34": {
            "inputs": {
                "text": prompt,
                "clip": ["30", 1]  # CLIP from checkpoint
            },
            "class_type": "CLIPTextEncode"
        },

        # 35. CLIP Text Encode (negative prompt)
        "35": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["30", 1]
            },
            "class_type": "CLIPTextEncode"
        },

        # 36. VAE Encode base image
        "36": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["30", 2]
            },
            "class_type": "VAEEncode"
        },

        # 37. Set Latent Noise Mask (for inpainting)
        "37": {
            "inputs": {
                "samples": ["36", 0],  # Latent from base
                "mask": ["2", 1]  # Mask image (alpha channel)
            },
            "class_type": "SetLatentNoiseMask"
        },

        # 38. KSampler (inpainting)
        "38": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": ["29", 0],  # IPAdapter model
                "positive": ["33", 0],  # ControlNet conditioning
                "negative": ["33", 1],
                "latent_image": ["37", 0]  # Masked latent
            },
            "class_type": "KSampler"
        },

        # 39. VAE Decode
        "39": {
            "inputs": {
                "samples": ["38", 0],
                "vae": ["30", 2]
            },
            "class_type": "VAEDecode"
        },

        # 40. Save Image
        "40": {
            "inputs": {
                "filename_prefix": "ipadapter_generated",
                "images": ["39", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow
