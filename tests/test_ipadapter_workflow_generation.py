"""Tests for IPAdapter workflow generation in generation script."""
import pytest
from pathlib import Path
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow


def test_workflow_generation_with_real_parameters():
    """Test that workflow can be generated with real parameters."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        prompt="Brown leather armor with shoulder pauldrons, chest plate, arm guards, leg armor, fantasy RPG character, pixel art style, detailed, high quality",
        negative_prompt="blurry, low quality, distorted, deformed, multiple heads, extra limbs, modern clothing",
        seed=42,
        steps=35,
        cfg=7.0,
        denoise=1.0
    )

    # Verify workflow structure
    assert isinstance(workflow, dict)
    assert len(workflow) > 0

    # Check that all 25 reference images are loaded
    load_image_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "LoadImage"
    ]
    # Should have: 1 base + 1 mask + 25 references = 27 LoadImage nodes
    assert len(load_image_nodes) == 27

    # Verify critical nodes exist
    node_types = [node.get("class_type") for node in workflow.values()]
    assert "IPAdapterModelLoader" in node_types
    assert "IPAdapterApply" in node_types
    assert "ControlNetLoader" in node_types
    assert "ControlNetApplyAdvanced" in node_types
    assert "OpenposePreprocessor" in node_types
    assert "SetLatentNoiseMask" in node_types
    assert "KSampler" in node_types
    assert "SaveImage" in node_types


def test_workflow_uses_correct_prompt():
    """Test that workflow uses the provided prompts."""
    test_prompt = "Test positive prompt"
    test_negative = "Test negative prompt"

    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        prompt=test_prompt,
        negative_prompt=test_negative,
        seed=42
    )

    # Find CLIP text encode nodes
    clip_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "CLIPTextEncode"
    ]

    assert len(clip_nodes) == 2

    prompts = [node["inputs"]["text"] for node in clip_nodes]
    assert test_prompt in prompts
    assert test_negative in prompts


def test_workflow_uses_correct_seed():
    """Test that workflow uses the provided seed."""
    test_seed = 12345

    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        seed=test_seed
    )

    # Find KSampler node
    ksampler_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "KSampler"
    ]

    assert len(ksampler_nodes) == 1
    assert ksampler_nodes[0]["inputs"]["seed"] == test_seed


def test_workflow_references_all_25_images():
    """Test that all 25 reference images are loaded."""
    reference_names = [f"clothed_frame_{i:02d}.png" for i in range(25)]

    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=reference_names,
        seed=42
    )

    # Find all LoadImage nodes that reference clothed frames
    load_image_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "LoadImage" and
           "clothed_frame" in node["inputs"].get("image", "")
    ]

    assert len(load_image_nodes) == 25

    # Verify all reference names are present
    loaded_images = [node["inputs"]["image"] for node in load_image_nodes]
    for ref_name in reference_names:
        assert ref_name in loaded_images
