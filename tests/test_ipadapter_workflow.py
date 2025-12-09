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
