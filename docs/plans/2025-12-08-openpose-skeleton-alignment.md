# OpenPose Skeleton-Based Frame Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix frame alignment by using OpenPose skeleton keypoint matching instead of broken bounding-box center alignment.

**Architecture:** Extract OpenPose body keypoints (neck, shoulders, hips) from both base and clothed frames using ComfyUI's ControlNet preprocessor. Calculate alignment offset by matching upper body center of mass. Apply transformation to align clothed frame to base frame position. Extend armor to cover feet. Extract final clothing spritesheet.

**Tech Stack:** ComfyUI (OpenPose ControlNet preprocessor), Python (numpy, PIL), existing sprite_clothing_gen infrastructure

---

## Prerequisites

**ComfyUI Server Must Be Running:**
```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
python main.py
```

Verify server is running at http://127.0.0.1:8188

**Required ComfyUI Components:**
- OpenPose ControlNet preprocessor (comfyui_controlnet_aux)
- Should already be installed from previous setup

---

### Task 1: Create OpenPose Keypoint Extraction Test

**Files:**
- Create: `tests/test_openpose_alignment.py`
- Uses: `sprite_clothing_gen/comfy_client.py`

**Step 1: Write failing test for keypoint extraction**

```python
"""Tests for OpenPose skeleton-based alignment."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from align_with_openpose import extract_openpose_keypoints, parse_keypoints


def test_extract_openpose_keypoints_returns_valid_structure():
    """Test that OpenPose extraction returns keypoints in expected format."""
    # Use existing test frame
    frame_path = Path("training_data/frames/base_frame_00.png")

    if not frame_path.exists():
        pytest.skip("Test frame not found")

    keypoints = extract_openpose_keypoints(str(frame_path))

    # Should return dict with 'people' array
    assert keypoints is not None
    assert 'people' in keypoints
    assert len(keypoints['people']) > 0

    # First person should have pose_keypoints_2d
    person = keypoints['people'][0]
    assert 'pose_keypoints_2d' in person

    # Should have 18 keypoints * 3 values (x, y, confidence)
    assert len(person['pose_keypoints_2d']) == 54


def test_parse_keypoints_extracts_upper_body_center():
    """Test that we correctly extract upper body center from keypoints."""
    # Mock keypoints for a person
    # OpenPose format: [x, y, conf] for each of 18 keypoints
    # Key indices: 1=neck, 2=right_shoulder, 5=left_shoulder
    mock_keypoints = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,  # 0: nose
                100, 100, 0.9,  # 1: neck
                90, 110, 0.9,  # 2: right shoulder
                0, 0, 0,  # 3: right elbow
                0, 0, 0,  # 4: right wrist
                110, 110, 0.9,  # 5: left shoulder
                # ... rest zeros
            ] + [0] * (48)  # Remaining keypoints
        }]
    }

    center_x, center_y = parse_keypoints(mock_keypoints)

    # Should average neck (100, 100), right shoulder (90, 110), left shoulder (110, 110)
    # Expected: x = (100 + 90 + 110) / 3 = 100, y = (100 + 110 + 110) / 3 ≈ 106.67
    assert abs(center_x - 100.0) < 0.1
    assert abs(center_y - 106.67) < 0.1
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
pytest tests/test_openpose_alignment.py::test_extract_openpose_keypoints_returns_valid_structure -v
```

Expected: FAIL with "module 'align_with_openpose' has no attribute 'extract_openpose_keypoints'"

**Step 3: Implement OpenPose keypoint extraction**

Create: `align_with_openpose.py`

```python
#!/usr/bin/env python3
"""Align clothed frames using OpenPose skeleton keypoint matching."""

import json
import requests
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional


def extract_openpose_keypoints(frame_path: str) -> Dict:
    """Extract OpenPose keypoints from a frame using ComfyUI.

    Args:
        frame_path: Path to frame image

    Returns:
        Dict with 'people' array containing pose_keypoints_2d
    """
    from sprite_clothing_gen.comfy_client import ComfyUIClient
    from sprite_clothing_gen.workflow_builder import build_openpose_preprocessing_workflow

    client = ComfyUIClient("http://127.0.0.1:8188")

    if not client.health_check():
        raise RuntimeError("ComfyUI server not running at http://127.0.0.1:8188")

    # Build workflow for OpenPose preprocessing
    workflow = build_openpose_preprocessing_workflow(frame_path)

    # Queue workflow
    prompt_id = client.queue_prompt(workflow)

    # Wait for completion and get results
    result = client.wait_for_completion(prompt_id, timeout=60)

    # Parse OpenPose JSON output from workflow results
    # The preprocessor should output JSON with keypoints
    if 'outputs' in result and result['outputs']:
        # Extract keypoints from first output
        for node_id, output in result['outputs'].items():
            if 'openpose_json' in output:
                return json.loads(output['openpose_json'][0])

    # Fallback: if no JSON in outputs, return empty structure
    return {'people': []}


def parse_keypoints(keypoints: Dict) -> Tuple[float, float]:
    """Extract upper body center from OpenPose keypoints.

    Uses neck, right shoulder, and left shoulder to calculate center.

    Args:
        keypoints: OpenPose keypoints dict with 'people' array

    Returns:
        (center_x, center_y) tuple
    """
    if not keypoints or 'people' not in keypoints or len(keypoints['people']) == 0:
        raise ValueError("No people detected in keypoints")

    person = keypoints['people'][0]
    kp = person['pose_keypoints_2d']

    # Extract key landmarks (index * 3 for x, index * 3 + 1 for y)
    # 1 = neck, 2 = right shoulder, 5 = left shoulder
    neck_x, neck_y = kp[1*3], kp[1*3+1]
    r_shoulder_x, r_shoulder_y = kp[2*3], kp[2*3+1]
    l_shoulder_x, l_shoulder_y = kp[5*3], kp[5*3+1]

    # Calculate center of mass
    center_x = (neck_x + r_shoulder_x + l_shoulder_x) / 3.0
    center_y = (neck_y + r_shoulder_y + l_shoulder_y) / 3.0

    return center_x, center_y
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_openpose_alignment.py -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add align_with_openpose.py tests/test_openpose_alignment.py
git commit -m "feat: add OpenPose keypoint extraction for alignment"
```

---

### Task 2: Implement Keypoint-Based Alignment Calculation

**Files:**
- Modify: `align_with_openpose.py`
- Modify: `tests/test_openpose_alignment.py`

**Step 1: Write failing test for alignment calculation**

Add to `tests/test_openpose_alignment.py`:

```python
def test_calculate_alignment_offset():
    """Test calculating alignment offset from two sets of keypoints."""
    from align_with_openpose import calculate_alignment_offset

    # Base frame keypoints (person centered at 100, 100)
    base_kp = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,  # nose
                100, 100, 0.9,  # neck
                90, 110, 0.9,  # right shoulder
                0, 0, 0, 0, 0, 0,  # elbow, wrist
                110, 110, 0.9,  # left shoulder
            ] + [0] * 42
        }]
    }

    # Clothed frame keypoints (person offset by +10, +5)
    clothed_kp = {
        'people': [{
            'pose_keypoints_2d': [
                0, 0, 0,
                110, 105, 0.9,  # neck at 110, 105
                100, 115, 0.9,  # right shoulder at 100, 115
                0, 0, 0, 0, 0, 0,
                120, 115, 0.9,  # left shoulder at 120, 115
            ] + [0] * 42
        }]
    }

    offset_x, offset_y = calculate_alignment_offset(base_kp, clothed_kp)

    # Base center: (100+90+110)/3 = 100, (100+110+110)/3 ≈ 106.67
    # Clothed center: (110+100+120)/3 = 110, (105+115+115)/3 ≈ 111.67
    # Offset: 100 - 110 = -10, 106.67 - 111.67 = -5
    assert offset_x == -10
    assert offset_y == -5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_openpose_alignment.py::test_calculate_alignment_offset -v
```

Expected: FAIL with "module has no attribute 'calculate_alignment_offset'"

**Step 3: Implement alignment calculation**

Add to `align_with_openpose.py`:

```python
def calculate_alignment_offset(base_keypoints: Dict, clothed_keypoints: Dict) -> Tuple[int, int]:
    """Calculate pixel offset needed to align clothed frame to base frame.

    Args:
        base_keypoints: OpenPose keypoints for base frame
        clothed_keypoints: OpenPose keypoints for clothed frame

    Returns:
        (offset_x, offset_y) tuple in pixels
    """
    base_center_x, base_center_y = parse_keypoints(base_keypoints)
    clothed_center_x, clothed_center_y = parse_keypoints(clothed_keypoints)

    # Calculate offset to move clothed center to base center
    offset_x = int(base_center_x - clothed_center_x)
    offset_y = int(base_center_y - clothed_center_y)

    return offset_x, offset_y
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_openpose_alignment.py::test_calculate_alignment_offset -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add align_with_openpose.py tests/test_openpose_alignment.py
git commit -m "feat: add keypoint-based alignment offset calculation"
```

---

### Task 3: Implement Image Transformation

**Files:**
- Modify: `align_with_openpose.py`
- Modify: `tests/test_openpose_alignment.py`

**Step 1: Write failing test for image transformation**

Add to `tests/test_openpose_alignment.py`:

```python
def test_apply_alignment_transform():
    """Test applying alignment offset to shift an image."""
    from align_with_openpose import apply_alignment_transform

    # Create test image: 100x100 white square in top-left corner
    img = np.zeros((200, 200, 4), dtype=np.uint8)
    img[0:100, 0:100] = [255, 255, 255, 255]  # White square

    # Shift by +50, +50
    offset_x, offset_y = 50, 50

    aligned = apply_alignment_transform(img, offset_x, offset_y)

    # White square should now be at 50:150, 50:150
    assert np.all(aligned[50:150, 50:150] == [255, 255, 255, 255])
    # Original position should be transparent
    assert np.all(aligned[0:50, 0:50, 3] == 0)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_openpose_alignment.py::test_apply_alignment_transform -v
```

Expected: FAIL with "module has no attribute 'apply_alignment_transform'"

**Step 3: Implement transformation**

Add to `align_with_openpose.py`:

```python
def apply_alignment_transform(img: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    """Apply alignment offset to shift an image.

    Args:
        img: RGBA image as numpy array (height, width, 4)
        offset_x: Horizontal offset in pixels
        offset_y: Vertical offset in pixels

    Returns:
        Aligned RGBA image as numpy array
    """
    height, width = img.shape[:2]
    aligned = np.zeros_like(img)

    # Calculate source and destination slices for the shift
    if offset_x >= 0 and offset_y >= 0:
        # Shift right and down
        dst_y = slice(offset_y, height)
        dst_x = slice(offset_x, width)
        src_y = slice(0, height - offset_y)
        src_x = slice(0, width - offset_x)
    elif offset_x >= 0 and offset_y < 0:
        # Shift right and up
        dst_y = slice(0, height + offset_y)
        dst_x = slice(offset_x, width)
        src_y = slice(-offset_y, height)
        src_x = slice(0, width - offset_x)
    elif offset_x < 0 and offset_y >= 0:
        # Shift left and down
        dst_y = slice(offset_y, height)
        dst_x = slice(0, width + offset_x)
        src_y = slice(0, height - offset_y)
        src_x = slice(-offset_x, width)
    else:
        # Shift left and up
        dst_y = slice(0, height + offset_y)
        dst_x = slice(0, width + offset_x)
        src_y = slice(-offset_y, height)
        src_x = slice(-offset_x, width)

    # Apply shift
    aligned[dst_y, dst_x] = img[src_y, src_x]

    return aligned
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_openpose_alignment.py::test_apply_alignment_transform -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add align_with_openpose.py tests/test_openpose_alignment.py
git commit -m "feat: add image alignment transformation"
```

---

### Task 4: Build Full Alignment Pipeline

**Files:**
- Modify: `align_with_openpose.py`
- Create: `tests/test_alignment_integration.py`

**Step 1: Write integration test**

Create `tests/test_alignment_integration.py`:

```python
"""Integration test for full alignment pipeline."""

import pytest
from pathlib import Path
from align_with_openpose import align_frame_with_openpose


@pytest.mark.integration
def test_align_frame_with_openpose():
    """Test full alignment pipeline on real frames."""
    base_path = Path("training_data/frames/base_frame_00.png")
    clothed_path = Path("training_data/frames/clothed_frame_00.png")

    if not base_path.exists() or not clothed_path.exists():
        pytest.skip("Test frames not found")

    # This will only pass if ComfyUI server is running
    try:
        aligned = align_frame_with_openpose(str(base_path), str(clothed_path))

        # Should return aligned image as numpy array
        assert aligned is not None
        assert aligned.shape == (512, 512, 4)  # RGBA
        assert aligned.dtype == 'uint8'

    except RuntimeError as e:
        if "ComfyUI server not running" in str(e):
            pytest.skip("ComfyUI server not running")
        raise
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_alignment_integration.py -v -m integration
```

Expected: FAIL with "module has no attribute 'align_frame_with_openpose'"

**Step 3: Implement full pipeline**

Add to `align_with_openpose.py`:

```python
def align_frame_with_openpose(base_frame_path: str, clothed_frame_path: str) -> np.ndarray:
    """Align clothed frame to base frame using OpenPose skeleton matching.

    Args:
        base_frame_path: Path to base frame
        clothed_frame_path: Path to clothed frame

    Returns:
        Aligned clothed frame as RGBA numpy array
    """
    # Extract keypoints from both frames
    print(f"  Extracting OpenPose keypoints from base frame...")
    base_kp = extract_openpose_keypoints(base_frame_path)

    print(f"  Extracting OpenPose keypoints from clothed frame...")
    clothed_kp = extract_openpose_keypoints(clothed_frame_path)

    # Calculate alignment offset
    offset_x, offset_y = calculate_alignment_offset(base_kp, clothed_kp)
    print(f"  Calculated offset: ({offset_x:+d}, {offset_y:+d})")

    # Load clothed frame
    clothed_img = np.array(Image.open(clothed_frame_path).convert('RGBA'))

    # Apply transformation
    aligned = apply_alignment_transform(clothed_img, offset_x, offset_y)

    return aligned
```

**Step 4: Run test to verify it passes (with ComfyUI running)**

```bash
# Make sure ComfyUI is running first!
pytest tests/test_alignment_integration.py -v -m integration
```

Expected: PASS (if ComfyUI is running) or SKIP (if not running)

**Step 5: Commit**

```bash
git add align_with_openpose.py tests/test_alignment_integration.py
git commit -m "feat: implement full OpenPose alignment pipeline"
```

---

### Task 5: Process All 25 Frames

**Files:**
- Modify: `align_with_openpose.py` (add main function)

**Step 1: Add main function to process all frames**

Add to `align_with_openpose.py`:

```python
def main():
    """Align all 25 clothed frames to base frames using OpenPose."""
    from sprite_clothing_gen.comfy_client import ComfyUIClient

    # Check ComfyUI is running
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        print("ERROR: ComfyUI server not running at http://127.0.0.1:8188")
        print("Start it with: cd /Users/roberthyatt/Code/ComfyUI && python main.py")
        return 1

    frames_dir = Path("training_data/frames")
    output_dir = Path("training_data/frames_aligned_openpose")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ALIGNING FRAMES USING OPENPOSE SKELETON MATCHING")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        clothed_path = frames_dir / f"clothed_frame_{frame_idx:02d}.png"

        # Align frame
        aligned = align_frame_with_openpose(str(base_path), str(clothed_path))

        # Save
        output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
        Image.fromarray(aligned).save(output_path)
        print(f"  ✓ Saved to {output_path}")
        print()

    print("=" * 70)
    print("✓ All frames aligned using OpenPose keypoints")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Step 2: Test manually on frame 00**

```bash
# Make sure ComfyUI is running!
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
python align_with_openpose.py
```

Expected: Processes frame 00 and saves aligned version

**Step 3: Verify alignment quality**

Create quick verification script:

```bash
python -c "
from PIL import Image
import numpy as np

# Load and compare
base = Image.open('training_data/frames/base_frame_00.png')
aligned = Image.open('training_data/frames_aligned_openpose/clothed_frame_00.png')
overlay = Image.alpha_composite(base.convert('RGBA'), aligned.convert('RGBA'))
overlay.save('training_data/openpose_alignment_test.png')
print('Saved verification overlay to training_data/openpose_alignment_test.png')
"
```

**Step 4: If alignment looks good, process remaining frames**

If frame 00 looks properly aligned, let the script continue for all 25 frames.

**Step 5: Commit**

```bash
git add align_with_openpose.py
git commit -m "feat: add main function to process all 25 frames"
```

---

### Task 6: Extend Armor to Cover Feet

**Files:**
- Create: `extend_armor_feet.py`
- Reuse existing logic from previous attempt

**Step 1: Copy existing armor extension logic**

The armor extension logic from `extend_armor_over_feet.py` was correct - it just operated on misaligned frames. Copy it to work with OpenPose-aligned frames:

```python
#!/usr/bin/env python3
"""Extend armor to cover feet in OpenPose-aligned frames."""

import numpy as np
from PIL import Image
from pathlib import Path


def extend_armor_to_cover_feet(clothed_frame: Image.Image, base_frame: Image.Image) -> Image.Image:
    """Extend brown armor downward to cover gray feet.

    For each column, extend armor from its bottom-most pixel down to cover
    any base character pixels below it.
    """
    clothed_arr = np.array(clothed_frame.convert('RGBA'))
    base_arr = np.array(base_frame.convert('RGBA'))

    height, width = clothed_arr.shape[:2]

    # For each column, extend armor downward
    for x in range(width):
        # Find bottom-most armor pixel
        armor_alpha = clothed_arr[:, x, 3]
        armor_present = armor_alpha > 128

        if not armor_present.any():
            continue

        armor_bottom_y = np.max(np.where(armor_present)[0])

        # Find bottom-most base pixel (where feet end)
        base_alpha = base_arr[:, x, 3]
        base_present = base_alpha > 128

        if not base_present.any():
            continue

        base_bottom_y = np.max(np.where(base_present)[0])

        # Extend armor if base extends below it
        if base_bottom_y > armor_bottom_y:
            armor_color = clothed_arr[armor_bottom_y, x].copy()

            for y in range(armor_bottom_y + 1, min(base_bottom_y + 1, height)):
                clothed_arr[y, x] = armor_color

    return Image.fromarray(clothed_arr)


def main():
    """Extend armor in all OpenPose-aligned frames."""
    frames_dir = Path("training_data/frames")
    aligned_dir = Path("training_data/frames_aligned_openpose")
    output_dir = Path("training_data/frames_complete_openpose")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXTENDING ARMOR TO COVER FEET")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        print(f"Frame {frame_idx:02d}:")

        base_path = frames_dir / f"base_frame_{frame_idx:02d}.png"
        aligned_path = aligned_dir / f"clothed_frame_{frame_idx:02d}.png"

        base = Image.open(base_path)
        aligned = Image.open(aligned_path)

        # Extend armor
        extended = extend_armor_to_cover_feet(aligned, base)

        # Save
        output_path = output_dir / f"clothed_frame_{frame_idx:02d}.png"
        extended.save(output_path)

        # Count pixels
        extended_arr = np.array(extended)
        pixels = np.sum(extended_arr[:, :, 3] > 128)
        print(f"  {pixels:6d} pixels")
        print(f"  ✓ Saved to {output_path}")
        print()

    print("=" * 70)
    print("✓ Armor extended for all frames")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

**Step 2: Run armor extension**

```bash
python extend_armor_feet.py
```

**Step 3: Commit**

```bash
git add extend_armor_feet.py
git commit -m "feat: extend armor to cover feet in aligned frames"
```

---

### Task 7: Extract Final Clothing Spritesheet

**Files:**
- Modify existing `extract_clothing_final.py` to use OpenPose-aligned frames

**Step 1: Update extraction script**

```python
#!/usr/bin/env python3
"""Extract final clothing spritesheet from OpenPose-aligned frames."""

import numpy as np
from PIL import Image
from pathlib import Path


def main():
    """Extract clothing spritesheet from complete OpenPose-aligned frames."""
    complete_dir = Path("training_data/frames_complete_openpose")
    output_path = Path("training_data/clothing_spritesheet_openpose.png")

    # Create 5x5 spritesheet
    frame_size = 512
    spritesheet = Image.new('RGBA', (frame_size * 5, frame_size * 5), (0, 0, 0, 0))

    print("=" * 70)
    print("EXTRACTING FINAL CLOTHING SPRITESHEET (OpenPose-aligned)")
    print("=" * 70)
    print()

    for frame_idx in range(25):
        # Load complete frame
        complete = np.array(Image.open(complete_dir / f"clothed_frame_{frame_idx:02d}.png").convert('RGBA'))

        # Convert to image
        clothing_img = Image.fromarray(complete)

        # Calculate position in spritesheet
        row = frame_idx // 5
        col = frame_idx % 5
        x = col * frame_size
        y = row * frame_size

        # Paste into spritesheet
        spritesheet.paste(clothing_img, (x, y), clothing_img)

        pixels = np.sum(complete[:, :, 3] > 128)
        print(f"Frame {frame_idx:02d}: {pixels:6d} clothing pixels")

    # Save
    spritesheet.save(output_path)

    print()
    print("=" * 70)
    print(f"✓ Final clothing spritesheet saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

**Step 2: Run extraction**

```bash
python extract_clothing_final.py
```

**Step 3: Create final verification overlay**

```bash
python -c "
from PIL import Image

base = Image.open('examples/input/base.png').convert('RGBA')
clothing = Image.open('training_data/clothing_spritesheet_openpose.png').convert('RGBA')
overlay = Image.alpha_composite(base, clothing)
overlay.save('training_data/final_openpose_overlay.png')
print('✓ Final overlay saved to training_data/final_openpose_overlay.png')
"
```

**Step 4: Visually verify alignment**

Open `training_data/final_openpose_overlay.png` and check:
- Frame 09 shoulders/torso align properly
- All frames have consistent alignment
- Armor covers feet (no gray bleed-through)

**Step 5: Commit**

```bash
git add extract_clothing_final.py
git commit -m "feat: extract final clothing spritesheet from OpenPose-aligned frames"
```

---

### Task 8: Create Unified Pipeline Script

**Files:**
- Create: `process_clothing_spritesheet.py`

**Goal:** Single command that runs the entire pipeline from start to finish.

**Step 1: Write unified pipeline script**

Create `process_clothing_spritesheet.py`:

```python
#!/usr/bin/env python3
"""Unified pipeline for clothing spritesheet generation.

Runs complete workflow:
1. Align clothed frames to base using OpenPose
2. Extend armor to cover feet
3. Open mask validation tool for user review
4. Extract final clothing spritesheet

Usage:
    python process_clothing_spritesheet.py
"""

import sys
from pathlib import Path
from align_with_openpose import main as align_main
from extend_armor_feet import main as extend_main
from validate_predicted_masks import main as validate_main
from extract_clothing_final import main as extract_main


def main():
    """Run complete clothing spritesheet generation pipeline."""
    print("=" * 70)
    print("CLOTHING SPRITESHEET GENERATION PIPELINE")
    print("=" * 70)
    print()
    print("This pipeline will:")
    print("  1. Align clothed frames to base using OpenPose skeleton matching")
    print("  2. Extend armor to cover feet")
    print("  3. Generate initial masks using trained U-Net model")
    print("  4. Open mask validation tool for manual review")
    print("  5. Extract final clothing spritesheet from validated masks")
    print()
    print("=" * 70)
    print()

    # Step 1: Align frames using OpenPose
    print("\n" + "=" * 70)
    print("STEP 1/5: Aligning frames with OpenPose")
    print("=" * 70 + "\n")

    result = align_main()
    if result != 0:
        print("ERROR: OpenPose alignment failed")
        return 1

    # Step 2: Extend armor to cover feet
    print("\n" + "=" * 70)
    print("STEP 2/5: Extending armor to cover feet")
    print("=" * 70 + "\n")

    result = extend_main()
    if result != 0:
        print("ERROR: Armor extension failed")
        return 1

    # Step 3: Generate initial masks with U-Net
    print("\n" + "=" * 70)
    print("STEP 3/5: Generating masks with trained U-Net model")
    print("=" * 70 + "\n")

    import subprocess
    result = subprocess.run([
        sys.executable,
        "predict_masks_with_model.py",
        "--frames-dir", "training_data/frames_complete_openpose",
        "--output-dir", "training_data_validation/masks_corrected"
    ])

    if result.returncode != 0:
        print("ERROR: Mask prediction failed")
        return 1

    # Step 4: Open mask validation tool
    print("\n" + "=" * 70)
    print("STEP 4/5: Opening mask validation tool")
    print("=" * 70 + "\n")
    print("Review and correct masks as needed...")
    print("Press Save to accept each mask and move to next frame")
    print()

    # Copy complete frames to validation directory
    import shutil
    val_frames = Path("training_data_validation/frames")
    val_frames.mkdir(parents=True, exist_ok=True)

    for i in range(25):
        src = Path(f"training_data/frames_complete_openpose/clothed_frame_{i:02d}.png")
        dst = val_frames / f"clothed_frame_{i:02d}.png"
        shutil.copy(src, dst)

    result = validate_main()
    if result != 0:
        print("ERROR: Mask validation failed or cancelled")
        return 1

    # Step 5: Extract final clothing spritesheet
    print("\n" + "=" * 70)
    print("STEP 5/5: Extracting final clothing spritesheet")
    print("=" * 70 + "\n")

    # Update extraction to use validated masks
    from extract_clothing_final import extract_with_validated_masks

    output_path = Path("training_data/clothing_spritesheet_final.png")
    extract_with_validated_masks(
        frames_dir=Path("training_data/frames_complete_openpose"),
        masks_dir=Path("training_data_validation/masks_corrected"),
        output_path=output_path
    )

    # Create final verification overlay
    from PIL import Image
    base = Image.open("examples/input/base.png").convert('RGBA')
    clothing = Image.open(output_path).convert('RGBA')
    overlay = Image.alpha_composite(base, clothing)
    overlay.save("training_data/final_verification.png")

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("Final deliverables:")
    print(f"  - Clothing spritesheet: {output_path}")
    print(f"  - Verification overlay: training_data/final_verification.png")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Update predict_masks_with_model.py to accept custom paths**

Modify `predict_masks_with_model.py` to accept command-line arguments:

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-dir', type=Path, default=Path('training_data/frames_complete'))
    parser.add_argument('--output-dir', type=Path, default=Path('training_data/masks_predicted'))
    args = parser.parse_args()

    # Use args.frames_dir and args.output_dir instead of hardcoded paths
    # ... rest of implementation
```

**Step 3: Update extract_clothing_final.py to support validated masks**

Add function to `extract_clothing_final.py`:

```python
def extract_with_validated_masks(frames_dir: Path, masks_dir: Path, output_path: Path):
    """Extract clothing using validated masks."""
    frame_size = 512
    spritesheet = Image.new('RGBA', (frame_size * 5, frame_size * 5), (0, 0, 0, 0))

    for frame_idx in range(25):
        clothed = np.array(Image.open(frames_dir / f'clothed_frame_{frame_idx:02d}.png').convert('RGBA'))
        mask = np.array(Image.open(masks_dir / f'mask_{frame_idx:02d}.png').convert('L'))

        # Apply mask
        clothing = clothed.copy()
        clothing[:, :, 3] = np.where(mask > 128, clothed[:, :, 3], 0)

        # Paste into spritesheet
        clothing_img = Image.fromarray(clothing)
        row = frame_idx // 5
        col = frame_idx % 5
        spritesheet.paste(clothing_img, (col * frame_size, row * frame_size), clothing_img)

    spritesheet.save(output_path)
    print(f"✓ Extracted clothing spritesheet to {output_path}")
```

**Step 4: Make script executable and test**

```bash
chmod +x process_clothing_spritesheet.py
python process_clothing_spritesheet.py
```

Expected: Runs through all 5 steps sequentially

**Step 5: Commit**

```bash
git add process_clothing_spritesheet.py
git commit -m "feat: add unified pipeline for clothing spritesheet generation"
```

---

## Verification Checklist

After completing all tasks:

- [ ] ComfyUI server started successfully
- [ ] All 25 frames processed with OpenPose
- [ ] Frame 09 alignment verified (shoulders/torso match base)
- [ ] Armor extended to cover feet in all frames
- [ ] Final spritesheet extracted
- [ ] Visual verification shows proper alignment across all frames
- [ ] All tests passing: `pytest tests/ -v`
- [ ] All changes committed

---

## Troubleshooting

**If OpenPose keypoint extraction fails:**
- Check ComfyUI logs for errors
- Verify ControlNet preprocessor is installed: `cd custom_nodes && ls | grep controlnet`
- Try running a simple OpenPose workflow manually in ComfyUI UI

**If alignment still looks wrong:**
- Check that keypoints are being detected (not all zeros)
- Visualize the OpenPose skeleton overlays to verify detection quality
- May need to adjust which keypoints are used for center calculation

**If armor extension creates artifacts:**
- Verify the aligned frames are correct first
- Check that base frame feet are being properly detected
- May need to adjust alpha threshold for foot detection

---

## Next Steps After Plan Execution

Once alignment is complete and verified:

1. Use validated masks tool to review alignment quality
2. Generate masks using trained U-Net model
3. Make corrections if needed
4. Extract final production clothing spritesheet
