"""Tests for ComfyUI workflow builder."""

import pytest
from sprite_clothing_gen.workflow_builder import build_clothing_generation_workflow
from sprite_clothing_gen.config import CHECKPOINT_MODEL, CONTROLNET_MODEL


def test_build_workflow_structure():
    """Test workflow has required nodes."""
    workflow = build_clothing_generation_workflow(
        pose_image_filename="pose.png",
        reference_image_filename="ref.png",
        seed=42
    )

    # Workflow should be a dict
    assert isinstance(workflow, dict)

    # Should contain nodes
    assert len(workflow) > 0

    # Should have checkpoint loader node
    checkpoint_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "CheckpointLoaderSimple"
    ]
    assert len(checkpoint_nodes) == 1
    assert CHECKPOINT_MODEL in str(checkpoint_nodes[0])

    # Should have ControlNet loader node
    controlnet_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "ControlNetLoader"
    ]
    assert len(controlnet_nodes) == 1

    # Should have KSampler node
    sampler_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "KSampler"
    ]
    assert len(sampler_nodes) == 1
    assert sampler_nodes[0]["inputs"]["seed"] == 42


def test_workflow_has_save_image():
    """Test workflow saves output image."""
    workflow = build_clothing_generation_workflow(
        pose_image_filename="pose.png",
        reference_image_filename="ref.png",
        seed=123
    )

    # Should have SaveImage node
    save_nodes = [
        node for node in workflow.values()
        if node.get("class_type") == "SaveImage"
    ]
    assert len(save_nodes) == 1
