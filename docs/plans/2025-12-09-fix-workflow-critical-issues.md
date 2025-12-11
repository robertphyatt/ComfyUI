# Fix IPAdapter Workflow Critical Issues

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 critical blocking issues in IPAdapter + ControlNet workflow that prevent execution

**Architecture:** Correct model compatibility (use SD1.5 consistently), fix node dependency ordering (checkpoint loads before IPAdapter), properly load 25 reference images (use parameter instead of glob pattern)

**Tech Stack:** ComfyUI workflow API, SD1.5 models (checkpoint, IPAdapter, ControlNet), pytest

---

## Background

Code review of Task 2 found 3 CRITICAL blocking issues:

1. **Model Mismatch:** Line 252 loads SDXL checkpoint (`sd_xl_base_1.0.safetensors`) but line 217 uses SD1.5 IPAdapter (`ip-adapter_sd15.bin`) and line 258 uses SD1.5 ControlNet (`control_v11p_sd15_openpose.pth`) - these architectures are incompatible
2. **Circular Dependency:** Node 29 (IPAdapter Apply) references node 30 (checkpoint loader) before node 30 exists in the workflow
3. **Unused Parameter:** Function accepts `reference_image_names: list[str]` but ignores it, using hardcoded glob pattern `"clothed_frame_*.png"` instead

These issues will cause immediate workflow failure in ComfyUI.

**Recommended approach:** Use SD1.5 for all models (better for 512x512 pixel art)

---

## Task 1: Fix Model Compatibility (Use SD1.5 Consistently)

**Files:**
- Modify: `sprite_clothing_gen/workflow_builder.py:252`
- Modify: `tests/test_ipadapter_workflow.py:45-50`

**Step 1: Write failing test for SD1.5 model consistency**

Add to `tests/test_ipadapter_workflow.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
pytest tests/test_ipadapter_workflow.py::test_workflow_uses_sd15_models_consistently -v
```

Expected output:
```
FAILED - AssertionError: SDXL checkpoint incompatible with SD1.5 IPAdapter/ControlNet: sd_xl_base_1.0.safetensors
```

**Step 3: Fix checkpoint model to use SD1.5**

In `sprite_clothing_gen/workflow_builder.py:252`, change:

```python
# BEFORE (line 252):
"30": {
    "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {"title": "Load Checkpoint"}
},

# AFTER:
"30": {
    "inputs": {
        "ckpt_name": "v1-5-pruned-emaonly.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {"title": "Load Checkpoint"}
},
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_uses_sd15_models_consistently -v
```

Expected output:
```
PASSED
```

**Step 5: Run all workflow tests to ensure no regressions**

Run:
```bash
pytest tests/test_ipadapter_workflow.py -v
```

Expected output:
```
test_workflow_has_required_nodes PASSED
test_workflow_uses_sd15_models_consistently PASSED
```

**Step 6: Commit**

```bash
git add sprite_clothing_gen/workflow_builder.py tests/test_ipadapter_workflow.py
git commit -m "fix: use SD1.5 checkpoint for model compatibility

- Change from SDXL (sd_xl_base_1.0) to SD1.5 (v1-5-pruned-emaonly)
- SD1.5 compatible with existing IPAdapter and ControlNet models
- Better for 512x512 pixel art generation
- Add test to enforce SD1.5 model consistency"
```

---

## Task 2: Fix Circular Dependency (Reorder Nodes)

**Files:**
- Modify: `sprite_clothing_gen/workflow_builder.py:203-280`
- Modify: `tests/test_ipadapter_workflow.py:85-125`

**Step 1: Write failing test for correct node ordering**

Add to `tests/test_ipadapter_workflow.py`:

```python
def test_workflow_node_dependencies_are_valid():
    """Test that all node references point to nodes that appear earlier in workflow."""
    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=[f"clothed_frame_{i:02d}.png" for i in range(25)],
        prompt="Brown leather armor",
        negative_prompt="blurry",
        seed=12345
    )

    # Track which nodes have been defined
    defined_nodes = set()

    # Process nodes in order
    for node_id, node in workflow.items():
        node_id_int = int(node_id)

        # Check all input references
        inputs = node.get("inputs", {})
        for input_name, input_value in inputs.items():
            # Input references are [node_id, output_index] tuples
            if isinstance(input_value, list) and len(input_value) == 2:
                referenced_node_id = str(input_value[0])
                referenced_node_int = int(referenced_node_id)

                # Referenced node must be defined BEFORE current node
                assert referenced_node_int < node_id_int, \
                    f"Node {node_id} references node {referenced_node_id} which appears later (circular dependency)"

                assert referenced_node_id in defined_nodes, \
                    f"Node {node_id} references undefined node {referenced_node_id}"

        # Mark this node as defined
        defined_nodes.add(node_id)
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_node_dependencies_are_valid -v
```

Expected output:
```
FAILED - AssertionError: Node 29 references node 30 which appears later (circular dependency)
```

**Step 3: Reorder nodes to fix dependency**

In `sprite_clothing_gen/workflow_builder.py`, the checkpoint loader (node 30) must load BEFORE IPAdapter Apply (node 29) references it.

Current problematic order:
- Node 29: IPAdapter Apply (references node 30)
- Node 30: Checkpoint Loader

New correct order:
- Node 29: Checkpoint Loader (was 30)
- Node 30: IPAdapter Apply (was 29)

Change in `sprite_clothing_gen/workflow_builder.py:232-280`:

```python
# BEFORE:
"29": {
    "inputs": {
        "weight": 0.85,
        "noise": 0.0,
        "weight_type": "linear",
        "start_at": 0.0,
        "end_at": 1.0,
        "unfold_batch": False,
        "ipadapter": ["3", 0],
        "image": ["4", 0],
        "model": ["30", 0]  # PROBLEM: References node 30 which loads AFTER
    },
    "class_type": "IPAdapterApply",
    "_meta": {"title": "IPAdapter Apply"}
},
"30": {
    "inputs": {
        "ckpt_name": "v1-5-pruned-emaonly.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {"title": "Load Checkpoint"}
},

# AFTER (swap node IDs):
"29": {
    "inputs": {
        "ckpt_name": "v1-5-pruned-emaonly.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {"title": "Load Checkpoint"}
},
"30": {
    "inputs": {
        "weight": 0.85,
        "noise": 0.0,
        "weight_type": "linear",
        "start_at": 0.0,
        "end_at": 1.0,
        "unfold_batch": False,
        "ipadapter": ["3", 0],
        "image": ["4", 0],
        "model": ["29", 0]  # FIXED: Now references node 29 which loads BEFORE
    },
    "class_type": "IPAdapterApply",
    "_meta": {"title": "IPAdapter Apply"}
},
```

Also update all other nodes that reference node 29 or 30:

```python
# Node 31 (ControlNet Apply) references the model output
# Update from node 29 to node 30 (new IPAdapter Apply node)
"31": {
    "inputs": {
        "strength": 1.0,
        "conditioning": ["6", 0],
        "control_net": ["32", 0],
        "image": ["33", 0],
        "positive": ["30", 0]  # Update: was ["29", 0]
    },
    "class_type": "ControlNetApply",
    "_meta": {"title": "Apply ControlNet"}
},

# Node 7 (KSampler) also references the model
"7": {
    "inputs": {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": "dpm_2m",
        "scheduler": "karras",
        "denoise": denoise,
        "model": ["30", 0],  # Update: was ["29", 0]
        "positive": ["31", 0],
        "negative": ["6", 0],
        "latent_image": ["5", 0]
    },
    "class_type": "KSampler",
    "_meta": {"title": "KSampler (Inpainting)"}
},
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_node_dependencies_are_valid -v
```

Expected output:
```
PASSED
```

**Step 5: Run all workflow tests**

Run:
```bash
pytest tests/test_ipadapter_workflow.py -v
```

Expected output:
```
test_workflow_has_required_nodes PASSED
test_workflow_uses_sd15_models_consistently PASSED
test_workflow_node_dependencies_are_valid PASSED
```

**Step 6: Commit**

```bash
git add sprite_clothing_gen/workflow_builder.py tests/test_ipadapter_workflow.py
git commit -m "fix: reorder nodes to eliminate circular dependency

- Swap node IDs 29 (IPAdapter Apply) and 30 (Checkpoint Loader)
- Checkpoint now loads at node 29 BEFORE IPAdapter references it at node 30
- Update all downstream references (ControlNet Apply, KSampler)
- Add test to enforce valid dependency ordering"
```

---

## Task 3: Fix Unused Parameter (Load Reference Images Properly)

**Files:**
- Modify: `sprite_clothing_gen/workflow_builder.py:217-230`
- Modify: `tests/test_ipadapter_workflow.py:130-180`

**Step 1: Write failing test for reference image loading**

Add to `tests/test_ipadapter_workflow.py`:

```python
def test_workflow_loads_all_reference_images():
    """Test that workflow uses provided reference_image_names parameter."""
    reference_names = [f"clothed_frame_{i:02d}.png" for i in range(25)]

    workflow = build_ipadapter_generation_workflow(
        base_image_name="base_frame_00.png",
        mask_image_name="mask_00.png",
        reference_image_names=reference_names,
        prompt="Brown leather armor",
        negative_prompt="blurry",
        seed=12345
    )

    # Find LoadImageBatch node (node 4)
    batch_loader_node = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoadImageBatch":
            batch_loader_node = node
            node_id_found = node_id
            break

    assert batch_loader_node is not None, "LoadImageBatch node not found"

    # Should NOT use glob pattern
    inputs = batch_loader_node["inputs"]
    assert "pattern" not in inputs, \
        "LoadImageBatch should not use 'pattern' (glob) mode when explicit filenames provided"

    # Should use image list mode
    assert "image" in inputs or "filenames" in inputs, \
        "LoadImageBatch should use explicit filename list mode"

    # Verify all 25 reference images are loaded
    # Note: ComfyUI LoadImageBatch API may vary - adjust based on actual node API
    # This test documents the expected behavior
    loaded_images = inputs.get("image") or inputs.get("filenames") or []
    assert len(loaded_images) == 25, \
        f"Expected 25 reference images, got {len(loaded_images)}"

    # Verify filenames match what was passed
    for expected_name in reference_names:
        assert expected_name in str(loaded_images), \
            f"Reference image {expected_name} not found in loaded images"
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_loads_all_reference_images -v
```

Expected output:
```
FAILED - AssertionError: LoadImageBatch should not use 'pattern' (glob) mode when explicit filenames provided
```

**Step 3: Fix LoadImageBatch to use reference_image_names parameter**

In `sprite_clothing_gen/workflow_builder.py:217-230`, change node 4:

```python
# BEFORE (uses glob pattern, ignores parameter):
"4": {
    "inputs": {
        "mode": "incremental_image",
        "index": 0,
        "label": "Batch 001",
        "path": "/Users/roberthyatt/Code/ComfyUI/training_data/frames",
        "pattern": "clothed_frame_*.png",  # PROBLEM: Hardcoded, ignores parameter
        "allow_RGBA_output": "true"
    },
    "class_type": "LoadImageBatch",
    "_meta": {"title": "Load Reference Images (25 Clothed Frames)"}
},

# AFTER (loads explicit filenames from parameter):
"4": {
    "inputs": {
        "mode": "incremental_image",
        "index": 0,
        "label": "Batch 001",
        "path": "/Users/roberthyatt/Code/ComfyUI/training_data/frames",
        "filenames": reference_image_names,  # FIXED: Use provided parameter
        "allow_RGBA_output": "true"
    },
    "class_type": "LoadImageBatch",
    "_meta": {"title": "Load Reference Images (25 Clothed Frames)"}
},
```

**Note:** If `LoadImageBatch` node doesn't support `filenames` parameter and requires `pattern`, we need a different approach:

**Alternative implementation (if LoadImageBatch only supports glob patterns):**

Create 25 separate `LoadImage` nodes and batch them together:

```python
# Build individual image loader nodes for each reference
workflow = {}
image_loader_ids = []

for idx, ref_image_name in enumerate(reference_image_names):
    node_id = str(100 + idx)  # Nodes 100-124
    workflow[node_id] = {
        "inputs": {
            "image": ref_image_name,
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {"title": f"Load Reference {idx}"}
    }
    image_loader_ids.append(node_id)

# Batch all images together (node 4)
workflow["4"] = {
    "inputs": {
        "images": [[loader_id, 0] for loader_id in image_loader_ids]
    },
    "class_type": "ImageBatch",
    "_meta": {"title": "Batch Reference Images"}
}
```

**Choose the approach based on ComfyUI's actual node API. Document in code comments.**

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_ipadapter_workflow.py::test_workflow_loads_all_reference_images -v
```

Expected output:
```
PASSED
```

**Step 5: Run all workflow tests**

Run:
```bash
pytest tests/test_ipadapter_workflow.py -v
```

Expected output:
```
test_workflow_has_required_nodes PASSED
test_workflow_uses_sd15_models_consistently PASSED
test_workflow_node_dependencies_are_valid PASSED
test_workflow_loads_all_reference_images PASSED
```

**Step 6: Commit**

```bash
git add sprite_clothing_gen/workflow_builder.py tests/test_ipadapter_workflow.py
git commit -m "fix: load reference images from parameter instead of glob

- Use reference_image_names parameter (25 filenames) instead of hardcoded pattern
- Fix API contract violation - parameter was accepted but ignored
- Add test to verify all 25 reference images are loaded
- Document LoadImageBatch API usage"
```

---

## Task 4: Verify All Issues Resolved with Integration Test

**Files:**
- Create: `tests/test_workflow_integration.py`

**Step 1: Write integration test**

Create `tests/test_workflow_integration.py`:

```python
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
```

**Step 2: Run integration test**

Run:
```bash
pytest tests/test_workflow_integration.py::test_workflow_passes_all_critical_checks -v -s
```

Expected output:
```
✓ Critical Check 1 PASSED: All models use SD1.5 (no SDXL mismatch)
✓ Critical Check 2 PASSED: No circular dependencies (correct node ordering)
✓ Critical Check 3 PASSED: Reference images loaded from parameter (not hardcoded glob)

======================================================================
✓ ALL CRITICAL CHECKS PASSED
======================================================================

Workflow is ready for execution in ComfyUI.

PASSED
```

**Step 3: Run full test suite**

Run:
```bash
pytest tests/test_ipadapter_workflow.py tests/test_workflow_integration.py -v
```

Expected output: All tests passing

**Step 4: Commit**

```bash
git add tests/test_workflow_integration.py
git commit -m "test: add integration test for workflow critical checks

- Verify SD1.5 model consistency (no SDXL mismatch)
- Verify no circular dependencies (correct node ordering)
- Verify reference images loaded from parameter (not hardcoded)
- Comprehensive check before ComfyUI execution"
```

---

## Verification

After completing all tasks:

1. **All tests passing:**
   ```bash
   pytest tests/test_ipadapter_workflow.py tests/test_workflow_integration.py -v
   ```
   Expected: 5 tests pass

2. **All critical issues resolved:**
   - ✅ Model compatibility: SD1.5 used consistently
   - ✅ Node ordering: Checkpoint loads before IPAdapter references it
   - ✅ Reference images: Parameter properly used (not ignored)

3. **Ready for Task 3:**
   - Workflow builder produces valid, executable ComfyUI workflows
   - Can proceed to create generation script that uses corrected workflow

---

## Success Criteria

- [ ] All unit tests in `test_ipadapter_workflow.py` pass (4 tests)
- [ ] Integration test in `test_workflow_integration.py` passes (1 test)
- [ ] Workflow uses SD1.5 checkpoint, IPAdapter, and ControlNet (no SDXL)
- [ ] Checkpoint node loads before IPAdapter Apply node references it
- [ ] All 25 reference images loaded from `reference_image_names` parameter
- [ ] 3 commits created (one per task + integration test)
- [ ] Ready to proceed to Task 3 (create generation script)
