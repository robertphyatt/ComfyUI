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
