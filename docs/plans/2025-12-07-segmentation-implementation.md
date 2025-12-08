# Semantic Segmentation + Edge Refinement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace bounding box approach with pixel-level semantic segmentation (AI) + edge-based boundary refinement (OpenCV) to eliminate armor over-removal.

**Architecture:** AI classifies pixels at 256×256 (clothing vs base character) using run-length encoding. Upscale to 512×512, detect edges on frame difference, snap rough AI boundaries to precise edges, apply morphological cleanup.

**Tech Stack:** Ollama ministral-3:8b (vision), OpenCV (Canny edge detection, morphology), PIL (image handling), NumPy (arrays)

---

## Task 1: Create Run-Length Encoding Helper Functions

**Files:**
- Create: `rle_utils.py`
- Test: `test_rle_utils.py`

**Step 1: Write the failing test**

```python
# test_rle_utils.py
import numpy as np
from rle_utils import encode_rle, decode_rle

def test_encode_rle_basic():
    """Test run-length encoding compresses consecutive values."""
    mask = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)
    rle = encode_rle(mask)

    assert rle == [
        {"value": 0, "count": 3},
        {"value": 1, "count": 2},
        {"value": 0, "count": 1}
    ]

def test_decode_rle_basic():
    """Test run-length decoding reconstructs original array."""
    rle = [
        {"value": 0, "count": 3},
        {"value": 1, "count": 2},
        {"value": 0, "count": 1}
    ]
    mask = decode_rle(rle, length=6)

    expected = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal(mask, expected)

def test_encode_decode_roundtrip_2d():
    """Test 2D mask survives encode->decode roundtrip."""
    original = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0]
    ], dtype=np.uint8)

    # Flatten, encode, decode, reshape
    flat = original.flatten()
    rle = encode_rle(flat)
    decoded_flat = decode_rle(rle, length=len(flat))
    decoded_2d = decoded_flat.reshape(original.shape)

    assert np.array_equal(decoded_2d, original)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_rle_utils.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'rle_utils'"

**Step 3: Write minimal implementation**

```python
# rle_utils.py
"""Run-length encoding utilities for mask compression."""
import numpy as np
from typing import List, Dict

def encode_rle(mask: np.ndarray) -> List[Dict[str, int]]:
    """Encode 1D binary mask as run-length encoded list.

    Args:
        mask: 1D numpy array of 0s and 1s

    Returns:
        List of {"value": int, "count": int} dicts

    Example:
        [0, 0, 0, 1, 1, 0] -> [{"value": 0, "count": 3}, {"value": 1, "count": 2}, {"value": 0, "count": 1}]
    """
    if len(mask) == 0:
        return []

    rle = []
    current_value = int(mask[0])
    current_count = 1

    for i in range(1, len(mask)):
        if mask[i] == current_value:
            current_count += 1
        else:
            rle.append({"value": current_value, "count": current_count})
            current_value = int(mask[i])
            current_count = 1

    # Append final run
    rle.append({"value": current_value, "count": current_count})

    return rle

def decode_rle(rle: List[Dict[str, int]], length: int) -> np.ndarray:
    """Decode run-length encoded list to 1D binary mask.

    Args:
        rle: List of {"value": int, "count": int} dicts
        length: Expected length of output array

    Returns:
        1D numpy array of 0s and 1s
    """
    mask = np.zeros(length, dtype=np.uint8)
    pos = 0

    for run in rle:
        value = run["value"]
        count = run["count"]
        mask[pos:pos + count] = value
        pos += count

    return mask
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_rle_utils.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add rle_utils.py test_rle_utils.py
git commit -m "feat: add run-length encoding utilities for mask compression"
```

---

## Task 2: Create AI Segmentation Function (with Mock)

**Files:**
- Create: `ai_segmentation.py`
- Test: `test_ai_segmentation.py`

**Step 1: Write the failing test with mock**

```python
# test_ai_segmentation.py
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from ai_segmentation import call_ollama_segmentation

def test_call_ollama_segmentation_basic():
    """Test AI segmentation calls Ollama and decodes RLE response."""
    # Create 256x256 test image
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    # Mock Ollama response with run-length encoding
    mock_response = {
        "response": '{"mask": [{"value": 1, "count": 65536}]}'  # All clothing
    }

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )

        mask = call_ollama_segmentation(clothed_256)

        # Verify shape and content
        assert mask.shape == (256, 256)
        assert np.all(mask == 1)  # All pixels classified as clothing

def test_call_ollama_segmentation_mixed():
    """Test AI segmentation handles mixed clothing/base regions."""
    clothed_256 = Image.new('RGB', (256, 256), color='brown')

    # Mock response: first 32k pixels = base (0), rest = clothing (1)
    mock_response = {
        "response": '{"mask": [{"value": 0, "count": 32768}, {"value": 1, "count": 32768}]}'
    }

    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: mock_response
        )

        mask = call_ollama_segmentation(clothed_256)

        # First half should be 0, second half should be 1
        assert mask.shape == (256, 256)
        flat = mask.flatten()
        assert np.all(flat[:32768] == 0)
        assert np.all(flat[32768:] == 1)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_ai_segmentation.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ai_segmentation'"

**Step 3: Write minimal implementation**

```python
# ai_segmentation.py
"""AI-powered semantic segmentation using Ollama vision model."""
import json
import base64
import io
import numpy as np
import requests
from PIL import Image
from typing import Dict, Any
from rle_utils import decode_rle

def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL image as base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_ollama_segmentation(clothed_frame_256: Image.Image) -> np.ndarray:
    """Call Ollama to generate semantic segmentation mask.

    Args:
        clothed_frame_256: 256x256 clothed character frame

    Returns:
        256x256 binary mask (0=base character, 1=clothing)
    """
    # Encode image
    image_b64 = encode_image_base64(clothed_frame_256)

    # Build prompt
    prompt = """Analyze this 256×256 pixel sprite showing a character wearing clothing/armor.

Classify each pixel as either:
- CLOTHING (1): New armor/clothing pixels (brown leather armor, equipment, etc.)
- BASE (0): Original gray character pixels showing through (gray skin/head visible through helmet)

Output as run-length encoding to compress the 65,536 pixel mask:
{
  "mask": [
    {"value": 0, "count": 1234},
    {"value": 1, "count": 567},
    ...
  ]
}

Rules:
- Start at top-left pixel, proceed row-by-row
- Group consecutive pixels with same value
- All runs must sum to exactly 65,536 pixels

Output ONLY valid JSON, no other text."""

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "ministral-3:8b",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }

    print("   Calling Ollama for semantic segmentation...")
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()

    result = response.json()
    response_text = result.get("response", "")

    # Parse RLE
    try:
        rle_data = json.loads(response_text)
        rle_mask = rle_data["mask"]
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to parse Ollama response: {e}\nResponse: {response_text}")

    # Decode RLE to flat array
    flat_mask = decode_rle(rle_mask, length=256*256)

    # Reshape to 2D
    mask_2d = flat_mask.reshape((256, 256))

    print(f"   ✓ Segmentation complete: {np.sum(mask_2d == 1)} clothing pixels, {np.sum(mask_2d == 0)} base pixels")

    return mask_2d
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_ai_segmentation.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add ai_segmentation.py test_ai_segmentation.py
git commit -m "feat: add AI semantic segmentation with RLE mask output"
```

---

## Task 3: Create Edge Detection Function

**Files:**
- Create: `edge_detection.py`
- Test: `test_edge_detection.py`

**Step 1: Write the failing test**

```python
# test_edge_detection.py
import numpy as np
from PIL import Image
import cv2
from edge_detection import detect_clothing_edges

def test_detect_clothing_edges_simple():
    """Test edge detection finds boundaries where clothing was added."""
    # Create base frame: gray character
    base_arr = np.full((512, 512, 4), [128, 128, 128, 255], dtype=np.uint8)
    base_frame = Image.fromarray(base_arr, 'RGBA')

    # Create clothed frame: add brown rectangle (clothing)
    clothed_arr = base_arr.copy()
    clothed_arr[200:300, 200:300] = [100, 70, 50, 255]  # Brown clothing
    clothed_frame = Image.fromarray(clothed_arr, 'RGBA')

    edges = detect_clothing_edges(base_frame, clothed_frame)

    # Should detect edges around the brown rectangle
    assert edges.shape == (512, 512)
    assert edges.dtype == np.uint8

    # Edges should exist near the rectangle boundaries
    # (Canny will detect the color transition)
    assert np.sum(edges[195:205, 195:205]) > 0  # Top-left corner region
    assert np.sum(edges[295:305, 295:305]) > 0  # Bottom-right corner region

def test_detect_clothing_edges_no_change():
    """Test edge detection finds no edges when frames are identical."""
    # Identical frames
    base_arr = np.full((512, 512, 4), [128, 128, 128, 255], dtype=np.uint8)
    base_frame = Image.fromarray(base_arr, 'RGBA')
    clothed_frame = Image.fromarray(base_arr, 'RGBA')

    edges = detect_clothing_edges(base_frame, clothed_frame)

    # No edges should be detected
    assert np.sum(edges) == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_edge_detection.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'edge_detection'"

**Step 3: Write minimal implementation**

```python
# edge_detection.py
"""Edge detection on frame differences for clothing boundary detection."""
import numpy as np
import cv2
from PIL import Image

def detect_clothing_edges(base_frame: Image.Image, clothed_frame: Image.Image) -> np.ndarray:
    """Detect edges where clothing was added (frame difference).

    Args:
        base_frame: Base character frame (512x512)
        clothed_frame: Clothed character frame (512x512)

    Returns:
        512x512 binary edge map (dilated by 1-2 pixels for search zones)
    """
    # Convert to grayscale
    base_arr = np.array(base_frame.convert('L'))
    clothed_arr = np.array(clothed_frame.convert('L'))

    # Compute absolute difference
    diff = cv2.absdiff(clothed_arr, base_arr)

    # Canny edge detection
    edges = cv2.Canny(diff, threshold1=50, threshold2=150)

    # Dilate edges to create search zones
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    print(f"   Edge detection: {np.sum(edges_dilated > 0)} edge pixels detected")

    return edges_dilated
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_edge_detection.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add edge_detection.py test_edge_detection.py
git commit -m "feat: add edge detection on frame differences for clothing boundaries"
```

---

## Task 4: Create Boundary Snapping Function

**Files:**
- Create: `boundary_snapping.py`
- Test: `test_boundary_snapping.py`

**Step 1: Write the failing test**

```python
# test_boundary_snapping.py
import numpy as np
from boundary_snapping import snap_mask_to_edges

def test_snap_mask_to_edges_basic():
    """Test snapping rough mask boundaries to detected edges."""
    # Create rough mask (blocky from upscaling)
    rough_mask = np.zeros((10, 10), dtype=np.uint8)
    rough_mask[4:7, 4:7] = 1  # 3x3 clothing region (blocky)

    # Create edge map with precise edge at x=5 (1 pixel right of rough boundary)
    edges = np.zeros((10, 10), dtype=np.uint8)
    edges[4:7, 5] = 255  # Vertical edge line

    refined_mask = snap_mask_to_edges(rough_mask, edges, search_radius=2)

    # Boundary should have moved right by 1 pixel to align with edge
    # (Implementation details will vary, but the mask should change)
    assert refined_mask.shape == (10, 10)
    assert not np.array_equal(refined_mask, rough_mask)  # Changed

def test_snap_mask_to_edges_no_edges():
    """Test snapping when no edges detected (should preserve rough mask)."""
    rough_mask = np.zeros((10, 10), dtype=np.uint8)
    rough_mask[4:7, 4:7] = 1

    edges = np.zeros((10, 10), dtype=np.uint8)  # No edges

    refined_mask = snap_mask_to_edges(rough_mask, edges, search_radius=2)

    # Should be unchanged
    assert np.array_equal(refined_mask, rough_mask)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_boundary_snapping.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'boundary_snapping'"

**Step 3: Write minimal implementation**

```python
# boundary_snapping.py
"""Snap rough AI mask boundaries to precise detected edges."""
import numpy as np
from scipy import ndimage

def snap_mask_to_edges(rough_mask: np.ndarray, edges: np.ndarray, search_radius: int = 10) -> np.ndarray:
    """Snap rough mask boundaries to detected edges.

    Args:
        rough_mask: 512x512 binary mask from AI (0=base, 1=clothing)
        edges: 512x512 edge map from Canny detection
        search_radius: Maximum distance to search for edges (pixels)

    Returns:
        512x512 refined binary mask with boundaries snapped to edges
    """
    refined_mask = rough_mask.copy()

    # Find boundary pixels (gradient magnitude > 0)
    # Using morphological gradient: difference between dilation and erosion
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = ndimage.binary_dilation(rough_mask, kernel)
    eroded = ndimage.binary_erosion(rough_mask, kernel)
    boundary = (dilated != eroded).astype(np.uint8)

    # Get coordinates of boundary pixels
    boundary_coords = np.argwhere(boundary > 0)

    if len(boundary_coords) == 0:
        return refined_mask

    # For each boundary pixel, find nearest edge within search_radius
    edge_coords = np.argwhere(edges > 0)

    if len(edge_coords) == 0:
        # No edges detected, return original
        return refined_mask

    snapped_count = 0

    for by, bx in boundary_coords:
        # Compute distances to all edge pixels
        distances = np.sqrt((edge_coords[:, 0] - by)**2 + (edge_coords[:, 1] - bx)**2)
        min_dist = np.min(distances)

        if min_dist <= search_radius:
            # Find nearest edge
            nearest_idx = np.argmin(distances)
            ey, ex = edge_coords[nearest_idx]

            # Snap: if edge is "outside" current mask region, expand
            # if edge is "inside" current mask region, contract
            if rough_mask[by, bx] == 1:
                # Clothing pixel on boundary
                # If nearest edge is at a base pixel, contract
                if ey < rough_mask.shape[0] and ex < rough_mask.shape[1]:
                    if rough_mask[ey, ex] == 0:
                        refined_mask[by, bx] = 0  # Contract
                        snapped_count += 1
            else:
                # Base pixel on boundary
                # If nearest edge is at a clothing pixel, expand
                if ey < rough_mask.shape[0] and ex < rough_mask.shape[1]:
                    if rough_mask[ey, ex] == 1:
                        refined_mask[by, bx] = 1  # Expand
                        snapped_count += 1

    print(f"   Boundary snapping: {snapped_count} pixels adjusted")

    return refined_mask
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_boundary_snapping.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add boundary_snapping.py test_boundary_snapping.py
git commit -m "feat: add boundary snapping to align AI mask with detected edges"
```

---

## Task 5: Create Morphological Cleanup Function

**Files:**
- Create: `morphological_cleanup.py`
- Test: `test_morphological_cleanup.py`

**Step 1: Write the failing test**

```python
# test_morphological_cleanup.py
import numpy as np
from morphological_cleanup import cleanup_mask

def test_cleanup_mask_fills_holes():
    """Test morphological cleanup fills small holes in clothing regions."""
    # Create mask with small hole in clothing region
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[5, 5] = 0  # Single pixel hole

    cleaned = cleanup_mask(mask)

    # Hole should be filled
    assert cleaned[5, 5] == 1

def test_cleanup_mask_removes_islands():
    """Test morphological cleanup removes isolated base character pixels."""
    # Create mask with isolated base pixel surrounded by clothing
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[5, 5] = 0  # Wait, this is same as above

    # Better test: isolated clothing pixel surrounded by base
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1  # Single pixel island

    cleaned = cleanup_mask(mask)

    # Island should be removed
    assert cleaned[5, 5] == 0

def test_cleanup_mask_preserves_large_regions():
    """Test morphological cleanup preserves large regions."""
    # Large clothing region
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1  # 10x10 region

    cleaned = cleanup_mask(mask)

    # Large region should be preserved
    assert np.sum(cleaned[5:15, 5:15]) > 80  # Most pixels preserved
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_morphological_cleanup.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'morphological_cleanup'"

**Step 3: Write minimal implementation**

```python
# morphological_cleanup.py
"""Morphological operations for mask cleanup."""
import numpy as np
import cv2

def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological operations to clean up mask.

    Operations:
    1. Close (fill small holes in clothing regions)
    2. Open (remove small isolated base character pixels)

    Args:
        mask: 512x512 binary mask (0=base, 1=clothing)

    Returns:
        512x512 cleaned binary mask
    """
    kernel = np.ones((3, 3), np.uint8)

    # Close operation: dilation followed by erosion
    # Fills small holes in clothing regions
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Open operation: erosion followed by dilation
    # Removes small isolated pixels
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    holes_filled = np.sum(mask_closed != mask)
    islands_removed = np.sum(mask_cleaned != mask_closed)

    print(f"   Morphological cleanup: {holes_filled} pixels filled, {islands_removed} islands removed")

    return mask_cleaned
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_morphological_cleanup.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add morphological_cleanup.py test_morphological_cleanup.py
git commit -m "feat: add morphological cleanup to fill holes and remove islands"
```

---

## Task 6: Integrate All Steps into New Extraction Function

**Files:**
- Create: `extract_clothing_segmentation.py`
- Test: `test_extract_clothing_segmentation.py`

**Step 1: Write the integration test (mocked AI)**

```python
# test_extract_clothing_segmentation.py
import numpy as np
from PIL import Image
from unittest.mock import patch
from extract_clothing_segmentation import extract_clothing_with_segmentation

def test_extract_clothing_with_segmentation_integration():
    """Test full segmentation pipeline integration."""
    # Create test frames
    base_arr = np.full((512, 512, 4), [128, 128, 128, 255], dtype=np.uint8)
    base_frame = Image.fromarray(base_arr, 'RGBA')

    clothed_arr = base_arr.copy()
    clothed_arr[200:400, 200:400] = [100, 70, 50, 255]  # Brown clothing region
    clothed_frame = Image.fromarray(clothed_arr, 'RGBA')

    # Mock AI segmentation to return simple mask
    # Top half = base (0), bottom half = clothing (1)
    mock_mask_256 = np.zeros((256, 256), dtype=np.uint8)
    mock_mask_256[128:, :] = 1

    with patch('extract_clothing_segmentation.call_ollama_segmentation', return_value=mock_mask_256):
        result = extract_clothing_with_segmentation(base_frame, clothed_frame)

        # Result should be RGBA image
        assert result.mode == 'RGBA'
        assert result.size == (512, 512)

        # Bottom half should have clothing, top half should be transparent
        result_arr = np.array(result)

        # Top half alpha should be mostly 0 (transparent)
        top_alpha = result_arr[:256, :, 3]
        assert np.mean(top_alpha) < 50  # Mostly transparent

        # Bottom half alpha should be mostly 255 (opaque)
        bottom_alpha = result_arr[256:, :, 3]
        assert np.mean(bottom_alpha) > 200  # Mostly opaque
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_extract_clothing_segmentation.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'extract_clothing_segmentation'"

**Step 3: Write the integration function**

```python
# extract_clothing_segmentation.py
"""Main clothing extraction using semantic segmentation + edge refinement."""
import numpy as np
import cv2
from PIL import Image

from ai_segmentation import call_ollama_segmentation
from edge_detection import detect_clothing_edges
from boundary_snapping import snap_mask_to_edges
from morphological_cleanup import cleanup_mask

def extract_clothing_with_segmentation(base_frame: Image.Image,
                                       clothed_frame: Image.Image) -> Image.Image:
    """Extract clothing using semantic segmentation + edge refinement.

    Pipeline:
    1. Downscale clothed frame to 256x256
    2. AI semantic segmentation (clothing vs base character)
    3. Upscale rough mask to 512x512
    4. Edge detection on frame difference
    5. Snap mask boundaries to detected edges
    6. Morphological cleanup
    7. Apply final mask

    Args:
        base_frame: Base character frame (512x512 RGBA)
        clothed_frame: Clothed character frame (512x512 RGBA)

    Returns:
        Clothing-only frame with transparent background (512x512 RGBA)
    """
    print("\n🧠 Semantic Segmentation + Edge Refinement Pipeline")

    # Step 1: Downscale to 256x256
    print("\n📐 Step 1: Downscaling to 256x256...")
    clothed_256 = clothed_frame.resize((256, 256), Image.LANCZOS)

    # Step 2: AI Semantic Segmentation
    print("\n🤖 Step 2: AI Semantic Segmentation...")
    mask_256 = call_ollama_segmentation(clothed_256)

    # Step 3: Upscale to 512x512
    print("\n📈 Step 3: Upscaling mask to 512x512...")
    mask_512_rough = cv2.resize(mask_256, (512, 512), interpolation=cv2.INTER_NEAREST)
    print(f"   Rough mask: {np.sum(mask_512_rough == 1)} clothing pixels, {np.sum(mask_512_rough == 0)} base pixels")

    # Step 4: Edge Detection
    print("\n🔍 Step 4: Detecting edges on frame difference...")
    edges = detect_clothing_edges(base_frame, clothed_frame)

    # Step 5: Boundary Snapping
    print("\n🎯 Step 5: Snapping boundaries to edges...")
    mask_512_refined = snap_mask_to_edges(mask_512_rough, edges, search_radius=10)

    # Step 6: Morphological Cleanup
    print("\n🧹 Step 6: Morphological cleanup...")
    mask_512_cleaned = cleanup_mask(mask_512_refined)

    # Step 7: Apply Mask
    print("\n✨ Step 7: Applying final mask...")
    clothed_arr = np.array(clothed_frame)
    clothing_arr = clothed_arr.copy()

    # Set alpha channel: mask==0 (base) → transparent, mask==1 (clothing) → keep
    clothing_arr[:, :, 3] = np.where(mask_512_cleaned == 1,
                                      clothed_arr[:, :, 3],  # Keep clothing alpha
                                      0)                     # Remove base character

    result = Image.fromarray(clothing_arr, 'RGBA')

    final_clothing_pixels = np.sum(clothing_arr[:, :, 3] > 0)
    print(f"   ✓ Final result: {final_clothing_pixels} clothing pixels preserved")

    return result
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_extract_clothing_segmentation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extract_clothing_segmentation.py test_extract_clothing_segmentation.py
git commit -m "feat: integrate segmentation pipeline - AI + edges + cleanup"
```

---

## Task 7: Create CLI Script for Testing Single Frame

**Files:**
- Create: `test_segmentation_single_frame.py`

**Step 1: Write the test script**

```python
# test_segmentation_single_frame.py
#!/usr/bin/env python3
"""Test segmentation pipeline on a single frame from debug_frames_ai/."""
import sys
from pathlib import Path
from PIL import Image
from extract_clothing_segmentation import extract_clothing_with_segmentation

def main():
    """Test segmentation on frame 0 (the problematic one)."""
    debug_dir = Path("debug_frames_ai")

    if not debug_dir.exists():
        print(f"Error: {debug_dir} not found. Run full extraction first to generate debug frames.")
        return 1

    # Load frame 0
    base_frame = Image.open(debug_dir / "frame_00_base.png")
    clothed_frame = Image.open(debug_dir / "frame_00_aligned.png")

    print("=" * 80)
    print("TESTING SEGMENTATION PIPELINE ON FRAME 0")
    print("Previous issue: Shoulder armor removed (white gaps)")
    print("Expected: Shoulders preserved, head removed")
    print("=" * 80)

    # Run segmentation
    result = extract_clothing_with_segmentation(base_frame, clothed_frame)

    # Save result
    output_path = debug_dir / "frame_00_segmentation_test.png"
    result.save(output_path)

    print()
    print(f"✓ Saved result to {output_path}")
    print()
    print("Visual inspection required:")
    print("1. Open the output image")
    print("2. Check shoulders: should be solid brown armor (no white gaps)")
    print("3. Check head: should be completely transparent")

    return 0

if __name__ == '__main__':
    sys.exit(main())
```

**Step 2: Run the test script (manual verification)**

Run: `python3 -u test_segmentation_single_frame.py`

Expected output:
```
================================================================================
TESTING SEGMENTATION PIPELINE ON FRAME 0
Previous issue: Shoulder armor removed (white gaps)
Expected: Shoulders preserved, head removed
================================================================================

🧠 Semantic Segmentation + Edge Refinement Pipeline

📐 Step 1: Downscaling to 256x256...

🤖 Step 2: AI Semantic Segmentation...
   Calling Ollama for semantic segmentation...
   ✓ Segmentation complete: 45123 clothing pixels, 20413 base pixels

...

✓ Saved result to debug_frames_ai/frame_00_segmentation_test.png
```

Manual verification: Open `debug_frames_ai/frame_00_segmentation_test.png` and visually inspect.

**Step 3: Commit**

```bash
git add test_segmentation_single_frame.py
git commit -m "test: add single frame segmentation test script"
```

---

## Task 8: Add --method Flag to Main Script for A/B Testing

**Files:**
- Modify: `extract_clothing_ai.py`

**Step 1: Add --method argument**

Add to argument parser (around line 570):

```python
parser.add_argument(
    '--method',
    type=str,
    choices=['bounding', 'segmentation'],
    default='bounding',
    help='Extraction method: "bounding" (AI bounding boxes + color matching) or "segmentation" (AI pixel segmentation + edge refinement). Default: bounding'
)
```

**Step 2: Import segmentation function at top of file**

```python
from extract_clothing_segmentation import extract_clothing_with_segmentation
```

**Step 3: Modify extraction calls to use selected method**

Find the extraction calls in Pass 1 (around line 630) and Pass 2 (around line 672). Wrap them:

```python
# Pass 1 (around line 630)
if args.method == 'segmentation':
    clothing_frame = extract_clothing_with_segmentation(base_frame, aligned_clothed)
    pixels_removed = 0  # Not tracked in segmentation method
else:
    # Existing bounding box approach
    clothing_frame, pixels_removed = extract_clothing_with_ai(
        base_frame, aligned_clothed, args.guidance, args.ai_guidance, i,
        tolerance=args.tolerance, max_retries=1, return_pixel_count=True
    )
```

Apply similar change to Pass 2 retry logic.

**Step 4: Test both methods**

Run bounding method:
```bash
python3 -u extract_clothing_ai.py \
  --base examples/input/base.png \
  --clothed examples/input/reference.png \
  --output examples/output/clothing_bounding.png \
  --method bounding \
  --debug
```

Run segmentation method:
```bash
python3 -u extract_clothing_ai.py \
  --base examples/input/base.png \
  --clothed examples/input/reference.png \
  --output examples/output/clothing_segmentation.png \
  --method segmentation \
  --debug
```

**Step 5: Commit**

```bash
git add extract_clothing_ai.py
git commit -m "feat: add --method flag to switch between bounding and segmentation approaches"
```

---

## Testing & Validation Checklist

After implementation, verify:

- [ ] Run `pytest -v` - all tests pass
- [ ] Run segmentation on frame 0 - shoulders preserved, head removed
- [ ] Run full 25-frame extraction with `--method segmentation`
- [ ] Compare bounding vs segmentation outputs visually
- [ ] Verify no white gaps in shoulder armor
- [ ] Verify gray head completely removed across all frames

## Implementation Complete

**Execution handoff:**

Plan complete and saved to `docs/plans/2025-12-07-segmentation-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
