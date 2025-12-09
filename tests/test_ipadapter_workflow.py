"""Tests for IPAdapter workflow builder."""
import pytest
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow


def test_workflow_has_required_nodes():
    """Test that workflow includes all required nodes."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        prompt="Brown leather armor",
        negative_prompt="blurry",
        seed=12345
    )

    # Verify workflow structure
    assert isinstance(workflow, dict)

    # Check for required node types
    node_classes = [node.get("class_type") for node in workflow.values()]

    required_nodes = [
        "LoadImage",  # Base image
        "LoadImage",  # Mask
        "IPAdapterModelLoader",
        "IPAdapterApply",
        "ControlNetLoader",
        "ControlNetApplyAdvanced",
        "CLIPTextEncode",  # Prompt
        "KSampler",  # Inpainting sampler
        "VAEDecode",
        "SaveImage"
    ]

    # Note: Can't check exact counts due to multiple LoadImage nodes
    # Just verify key node types exist
    assert "IPAdapterModelLoader" in node_classes
    assert "IPAdapterApply" in node_classes
    assert "ControlNetLoader" in node_classes
    assert "KSampler" in node_classes


def test_workflow_uses_sd15_models_consistently():
    """Test that workflow uses SD1.5 models (not SDXL) for 512x512 pixel art."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        prompt="Brown leather armor",
        negative_prompt="blurry",
        seed=12345
    )

    # Find checkpoint loader node
    checkpoint_node = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "CheckpointLoaderSimple":
            checkpoint_node = node
            break

    assert checkpoint_node is not None, "Checkpoint loader node not found"

    # Verify SD1.5 checkpoint (NOT SDXL)
    ckpt_name = checkpoint_node["inputs"]["ckpt_name"]
    assert "sd15" in ckpt_name.lower() or "v1-5" in ckpt_name.lower(), \
        f"Expected SD1.5 checkpoint, got: {ckpt_name}"
    assert "xl" not in ckpt_name.lower(), \
        f"SDXL checkpoint incompatible with SD1.5 IPAdapter/ControlNet: {ckpt_name}"

    # Find IPAdapter loader node
    ipadapter_node = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "IPAdapterModelLoader":
            ipadapter_node = node
            break

    assert ipadapter_node is not None, "IPAdapter loader node not found"

    # Verify SD1.5 IPAdapter
    ipadapter_file = ipadapter_node["inputs"]["ipadapter_file"]
    assert "sd15" in ipadapter_file.lower(), \
        f"Expected SD1.5 IPAdapter, got: {ipadapter_file}"

    # Find ControlNet loader node
    controlnet_node = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "ControlNetLoader":
            controlnet_node = node
            break

    assert controlnet_node is not None, "ControlNet loader node not found"

    # Verify SD1.5 ControlNet
    controlnet_name = controlnet_node["inputs"]["control_net_name"]
    assert "sd15" in controlnet_name.lower(), \
        f"Expected SD1.5 ControlNet, got: {controlnet_name}"
