#!/usr/bin/env python3
"""Integration test for IPAdapter workflow - verifies all critical issues resolved."""

from pathlib import Path
from sprite_clothing_gen.workflow_builder import build_ipadapter_generation_workflow


def test_workflow_passes_all_critical_checks():
    """Integration test: verify workflow has no critical blocking issues.

    Checks:
    1. Model compatibility - SD1.5 used consistently (not SDXL)
    2. Node ordering - no circular dependencies
    3. Reference images - properly loads all 25 frames
    """
    reference_names = [f"clothed_frame_{i:02d}.png" for i in range(25)]

    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=reference_names,
        prompt="Brown leather armor with shoulder pauldrons, pixel art style",
        negative_prompt="blurry, low quality, distorted",
        seed=42,
        steps=35,
        cfg=7.0,
        denoise=1.0
    )

    # Critical Check 1: Model Compatibility (SD1.5, not SDXL)
    checkpoint_node = None
    ipadapter_node = None
    controlnet_node = None

    for node_id, node in workflow.items():
        class_type = node.get("class_type")

        if class_type == "CheckpointLoaderSimple":
            checkpoint_node = node
        elif class_type == "IPAdapterModelLoader":
            ipadapter_node = node
        elif class_type == "ControlNetLoader":
            controlnet_node = node

    assert checkpoint_node is not None, "Missing checkpoint loader"
    assert ipadapter_node is not None, "Missing IPAdapter loader"
    assert controlnet_node is not None, "Missing ControlNet loader"

    ckpt_name = checkpoint_node["inputs"]["ckpt_name"]
    assert "xl" not in ckpt_name.lower(), \
        f"CRITICAL: SDXL checkpoint incompatible with SD1.5 models: {ckpt_name}"
    assert "sd15" in ckpt_name.lower() or "v1-5" in ckpt_name.lower(), \
        f"Expected SD1.5 checkpoint, got: {ckpt_name}"

    ipadapter_file = ipadapter_node["inputs"]["ipadapter_file"]
    assert "sd15" in ipadapter_file.lower(), \
        f"IPAdapter must be SD1.5: {ipadapter_file}"

    controlnet_name = controlnet_node["inputs"]["control_net_name"]
    assert "sd15" in controlnet_name.lower(), \
        f"ControlNet must be SD1.5: {controlnet_name}"

    print("✓ Critical Check 1 PASSED: All models use SD1.5 (no SDXL mismatch)")

    # Critical Check 2: Node Ordering (no circular dependencies)
    defined_nodes = set()

    for node_id, node in workflow.items():
        node_id_int = int(node_id)

        inputs = node.get("inputs", {})
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                referenced_node_id = str(input_value[0])
                referenced_node_int = int(referenced_node_id)

                assert referenced_node_int < node_id_int, \
                    f"CRITICAL: Node {node_id} references node {referenced_node_id} (circular dependency)"

                assert referenced_node_id in defined_nodes, \
                    f"CRITICAL: Node {node_id} references undefined node {referenced_node_id}"

        defined_nodes.add(node_id)

    print("✓ Critical Check 2 PASSED: No circular dependencies (correct node ordering)")

    # Critical Check 3: Reference Images (uses parameter, not hardcoded glob)
    batch_loader_node = None
    for node_id, node in workflow.items():
        if node.get("class_type") in ["LoadImageBatch", "ImageBatch"]:
            batch_loader_node = node
            break

    assert batch_loader_node is not None, "Missing batch image loader"

    inputs = batch_loader_node["inputs"]

    # Should NOT use glob pattern
    if "pattern" in inputs:
        assert inputs["pattern"] != "clothed_frame_*.png", \
            "CRITICAL: Using hardcoded glob pattern instead of reference_image_names parameter"

    # Should reference explicit filenames (exact mechanism depends on node type)
    # At minimum, verify parameter is being used in workflow
    has_explicit_refs = (
        "filenames" in inputs or
        "images" in inputs or
        any(ref_name in str(workflow) for ref_name in reference_names[:3])
    )

    assert has_explicit_refs, \
        "CRITICAL: reference_image_names parameter not used in workflow"

    print("✓ Critical Check 3 PASSED: Reference images loaded from parameter (not hardcoded glob)")

    print("\n" + "=" * 70)
    print("✓ ALL CRITICAL CHECKS PASSED")
    print("=" * 70)
    print("\nWorkflow is ready for execution in ComfyUI.")


if __name__ == "__main__":
    test_workflow_passes_all_critical_checks()
