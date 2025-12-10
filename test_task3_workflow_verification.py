#!/usr/bin/env python3
"""Verify Task 3 workflow has winning configuration: OpenPose + txt2img."""

from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow

def test_workflow_has_openpose():
    """Verify workflow uses ControlNet OpenPose."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)]
    )

    # Find ControlNet loader node
    controlnet_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "ControlNetLoader"
    ]

    assert len(controlnet_nodes) == 1, f"Expected 1 ControlNetLoader, found {len(controlnet_nodes)}"

    node_id, node_data = controlnet_nodes[0]
    control_net_name = node_data["inputs"]["control_net_name"]

    assert "openpose" in control_net_name.lower(), \
        f"Expected OpenPose ControlNet, got: {control_net_name}"

    print(f"✓ ControlNet: {control_net_name}")


def test_workflow_has_openpose_preprocessor():
    """Verify workflow uses OpenposePreprocessor."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)]
    )

    # Find OpenposePreprocessor node
    openpose_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "OpenposePreprocessor"
    ]

    assert len(openpose_nodes) == 1, f"Expected 1 OpenposePreprocessor, found {len(openpose_nodes)}"

    print(f"✓ OpenposePreprocessor present")


def test_workflow_uses_empty_latent_not_vae_encode():
    """Verify workflow uses EmptyLatentImage (txt2img), NOT VAEEncode (img2img)."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)]
    )

    # Check for EmptyLatentImage
    empty_latent_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "EmptyLatentImage"
    ]

    assert len(empty_latent_nodes) == 1, \
        f"Expected 1 EmptyLatentImage, found {len(empty_latent_nodes)}"

    # Verify NO VAEEncode nodes
    vae_encode_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "VAEEncode"
    ]

    assert len(vae_encode_nodes) == 0, \
        f"Expected 0 VAEEncode nodes (txt2img approach), found {len(vae_encode_nodes)}"

    print(f"✓ Using EmptyLatentImage (txt2img)")
    print(f"✓ NOT using VAEEncode (img2img)")


def test_workflow_does_not_use_set_latent_noise_mask():
    """Verify workflow does NOT use SetLatentNoiseMask (removed for txt2img)."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)]
    )

    # Verify NO SetLatentNoiseMask nodes
    set_mask_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "SetLatentNoiseMask"
    ]

    assert len(set_mask_nodes) == 0, \
        f"Expected 0 SetLatentNoiseMask nodes (txt2img doesn't use masks), found {len(set_mask_nodes)}"

    print(f"✓ NOT using SetLatentNoiseMask (txt2img approach)")


def test_workflow_ksampler_uses_denoise_1_0():
    """Verify KSampler default denoise is 1.0 for txt2img."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)]
    )

    # Find KSampler node
    ksampler_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "KSampler"
    ]

    assert len(ksampler_nodes) == 1, f"Expected 1 KSampler, found {len(ksampler_nodes)}"

    node_id, node_data = ksampler_nodes[0]
    denoise = node_data["inputs"]["denoise"]

    assert denoise == 1.0, f"Expected denoise=1.0 for txt2img, got {denoise}"

    print(f"✓ KSampler denoise: {denoise}")


def test_workflow_ksampler_latent_source():
    """Verify KSampler gets latent from EmptyLatentImage."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)]
    )

    # Find KSampler node
    ksampler_nodes = [
        (node_id, node_data)
        for node_id, node_data in workflow.items()
        if node_data.get("class_type") == "KSampler"
    ]

    assert len(ksampler_nodes) == 1
    ksampler_id, ksampler_data = ksampler_nodes[0]

    # Get the node ID that KSampler's latent_image comes from
    latent_source = ksampler_data["inputs"]["latent_image"]
    assert isinstance(latent_source, list), "latent_image should be [node_id, output_index]"

    latent_source_node_id = latent_source[0]
    latent_source_node = workflow[latent_source_node_id]

    assert latent_source_node["class_type"] == "EmptyLatentImage", \
        f"KSampler latent should come from EmptyLatentImage, got {latent_source_node['class_type']}"

    print(f"✓ KSampler latent_image source: EmptyLatentImage (node {latent_source_node_id})")


if __name__ == "__main__":
    print("=" * 70)
    print("VERIFYING TASK 3 WORKFLOW CONFIGURATION")
    print("Expected: ControlNet OpenPose + txt2img (EmptyLatentImage, denoise=1.0)")
    print("=" * 70)
    print()

    try:
        test_workflow_has_openpose()
        test_workflow_has_openpose_preprocessor()
        test_workflow_uses_empty_latent_not_vae_encode()
        test_workflow_does_not_use_set_latent_noise_mask()
        test_workflow_ksampler_uses_denoise_1_0()
        test_workflow_ksampler_latent_source()

        print()
        print("=" * 70)
        print("✓ ALL VERIFICATION TESTS PASSED")
        print("Workflow has winning configuration: OpenPose + txt2img")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ VERIFICATION FAILED: {e}")
        print("=" * 70)
        exit(1)
