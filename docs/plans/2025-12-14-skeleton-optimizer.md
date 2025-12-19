# Skeleton Keypoint Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an optimizer that adjusts clothed sprite skeleton keypoints to minimize visible gray (base mannequin) pixels in the final warped composite.

**Architecture:** Greedy hill-climbing optimizer that perturbs keypoints 1px at a time, evaluates the full warp pipeline, and keeps changes that reduce uncovered gray pixels. Stops when no improvements possible or zero gray remaining.

**Tech Stack:** Python, OpenCV, NumPy, SciPy (RBFInterpolator for TPS)

---

## Context

### Problem
When warping armored sprites to match base mannequin poses, gray pixels from the base "peek through" at hands and shoulders because:
1. Skeleton keypoints on clothed frames may be slightly off
2. No keypoints exist for fingertips
3. TPS warping can distort armor in ways that expose the base

### Solution
Optimize the clothed frame keypoints by directly minimizing visible gray in the final composite.

### Key Files
- **Annotations:** `/Users/roberthyatt/Code/ComfyUI/training_data/annotations.json` - dict keyed by filename, each entry has `image` and `keypoints` (dict of name -> [x, y])
- **Frames:** `/Users/roberthyatt/Code/ComfyUI/training_data/frames/` - base_frame_XX.png and clothed_frame_XX.png
- **Masks:** `/Users/roberthyatt/Code/ComfyUI/training_data/masks_corrected/mask_XX.png` - white = armor, black = body
- **Output:** `/Users/roberthyatt/Code/ComfyUI/training_data/skeleton_comparison/`
- **Existing module:** `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/`

### Constraints from User
1. **DO NOT** optimize head (index 0) or neck (index 1)
2. **Only** frames clothed_frame_00 and clothed_frame_01
3. **Move 1px** per test iteration
4. **Greedy stopping:** improvement = continue, same/worse = stop

### Frame Mappings (established in prior work)
- clothed_frame_00 → base_frame_23 (best pose match)
- clothed_frame_01 → base_frame_24 (best pose match)

### Scale Factor
- SCALE_FACTOR = 1.057 (clothed sprites are ~6% smaller)

---

## Task 1: Create optimizer.py module scaffold

**Files:**
- Create: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Create module with imports and config**

```python
"""Skeleton keypoint optimizer for minimizing gray pixel leakage."""

import json
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS


@dataclass
class OptimizerConfig:
    """Configuration for skeleton optimizer."""
    scale_factor: float = 1.057
    optimize_indices: Tuple[int, ...] = tuple(range(2, 14))  # Skip head(0), neck(1)
    step_size: int = 1
    max_iterations: int = 100


# Frame mappings (clothed -> best matching base)
FRAME_MAPPINGS = {
    "clothed_frame_00": "base_frame_23",
    "clothed_frame_01": "base_frame_24",
}
```

**Step 2: Verify module imports**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import OptimizerConfig, FRAME_MAPPINGS; print('OK')"`
Expected: `OK`

---

## Task 2: Add annotation loading helper

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add load_annotations function**

Add after the FRAME_MAPPINGS definition:

```python
def load_annotations(annotations_path: Path) -> Dict[str, Dict]:
    """Load annotations from JSON file."""
    with open(annotations_path) as f:
        return json.load(f)


def get_keypoints_array(annotations: Dict, frame_name: str) -> np.ndarray:
    """Get keypoints as numpy array [14, 2] for a frame.

    Args:
        annotations: Dict keyed by filename
        frame_name: Frame name without .png extension

    Returns:
        np.ndarray of shape [14, 2] with x, y coordinates
    """
    key = f"{frame_name}.png"
    if key not in annotations:
        raise KeyError(f"Frame {key} not found in annotations")

    kpts = annotations[key]["keypoints"]
    result = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)

    for i, name in enumerate(KEYPOINT_NAMES):
        if name in kpts:
            result[i] = kpts[name]

    return result
```

**Step 2: Test annotation loading**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -c "
from sprite_keypoint_detector.optimizer import load_annotations, get_keypoints_array
from pathlib import Path
ann = load_annotations(Path('training_data/annotations.json'))
kpts = get_keypoints_array(ann, 'base_frame_00')
print(f'Shape: {kpts.shape}')
print(f'Head: {kpts[0]}')
print(f'Neck: {kpts[1]}')
"
```
Expected: Shape [14, 2], coordinates for head and neck printed

---

## Task 3: Add TPS warp function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add thin_plate_spline_warp function**

Add after get_keypoints_array:

```python
def thin_plate_spline_warp(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> np.ndarray:
    """Warp image so src_points move to dst_points using TPS.

    Args:
        image: RGBA image as numpy array [H, W, 4]
        src_points: Source keypoints [N, 2]
        dst_points: Destination keypoints [N, 2]

    Returns:
        Warped RGBA image
    """
    h, w = image.shape[:2]

    # Add corner anchors to prevent edge distortion
    corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float64)
    src_all = np.vstack([src_points, corners])
    dst_all = np.vstack([dst_points, corners])

    # Build RBF interpolators for x and y mapping
    rbf_x = RBFInterpolator(dst_all, src_all[:, 0], kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(dst_all, src_all[:, 1], kernel='thin_plate_spline', smoothing=0)

    # Create coordinate grid for all pixels
    yy, xx = np.mgrid[0:h, 0:w]
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute source coordinates for each destination pixel
    map_x = rbf_x(grid_points).reshape(h, w).astype(np.float32)
    map_y = rbf_y(grid_points).reshape(h, w).astype(np.float32)

    # Warp image
    warped = cv2.remap(
        image, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped
```

**Step 2: Test TPS warp compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import thin_plate_spline_warp; print('OK')"`
Expected: `OK`

---

## Task 4: Add scale-and-align function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add scale_and_align_image function**

Add after thin_plate_spline_warp:

```python
def scale_and_align_image(
    image: np.ndarray,
    image_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    scale_factor: float,
    canvas_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale image and align by neck position.

    Args:
        image: Source RGBA image
        image_keypoints: Keypoints for source image [14, 2]
        target_keypoints: Target keypoints to align to [14, 2]
        scale_factor: Scale multiplier for source image
        canvas_size: Output canvas size

    Returns:
        Tuple of (scaled_aligned_image, scaled_aligned_keypoints)
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Scale image
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_kpts = image_keypoints * scale_factor

    # Calculate neck alignment offset (neck is index 1)
    neck_offset = target_keypoints[1] - scaled_kpts[1]
    offset_x = int(round(neck_offset[0]))
    offset_y = int(round(neck_offset[1]))

    # Create output canvas
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    # Calculate valid regions for copy
    src_x1 = max(0, -offset_x)
    src_x2 = min(new_w, canvas_size - offset_x)
    src_y1 = max(0, -offset_y)
    src_y2 = min(new_h, canvas_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = max(0, offset_y)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Copy to canvas
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

    # Adjust keypoints
    aligned_kpts = scaled_kpts + neck_offset

    return canvas, aligned_kpts
```

**Step 2: Test function compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import scale_and_align_image; print('OK')"`
Expected: `OK`

---

## Task 5: Add gray pixel counting function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add count_uncovered_pixels function**

Add after scale_and_align_image:

```python
def count_uncovered_pixels(base_image: np.ndarray, armor_image: np.ndarray) -> int:
    """Count base pixels not covered by armor.

    Args:
        base_image: Base mannequin RGBA image
        armor_image: Warped armor RGBA image (same size)

    Returns:
        Number of base pixels visible through armor
    """
    # Where base has content (alpha > 128)
    base_visible = base_image[:, :, 3] > 128

    # Where armor covers (alpha > 128)
    armor_covers = armor_image[:, :, 3] > 128

    # Uncovered = base visible AND armor doesn't cover
    uncovered = base_visible & ~armor_covers

    return int(np.sum(uncovered))
```

**Step 2: Test function compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import count_uncovered_pixels; print('OK')"`
Expected: `OK`

---

## Task 6: Add full pipeline evaluation function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add evaluate_keypoints function**

Add after count_uncovered_pixels:

```python
def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to image, setting alpha from mask.

    Args:
        image: RGBA image
        mask: Grayscale mask (white = keep)

    Returns:
        Masked RGBA image
    """
    result = image.copy()
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    result[:, :, 3] = np.minimum(result[:, :, 3], mask)
    return result


def evaluate_keypoints(
    clothed_image: np.ndarray,
    clothed_keypoints: np.ndarray,
    base_image: np.ndarray,
    base_keypoints: np.ndarray,
    mask_image: np.ndarray,
    config: OptimizerConfig
) -> Tuple[int, np.ndarray]:
    """Run full pipeline and return uncovered pixel count.

    Args:
        clothed_image: Original clothed sprite RGBA
        clothed_keypoints: Current keypoint estimates [14, 2]
        base_image: Target base sprite RGBA
        base_keypoints: Target keypoints [14, 2]
        mask_image: Armor mask (white = armor)
        config: Optimizer configuration

    Returns:
        Tuple of (uncovered_count, warped_composite)
    """
    # Scale and align clothed image
    scaled_clothed, scaled_kpts = scale_and_align_image(
        clothed_image,
        clothed_keypoints,
        base_keypoints,
        config.scale_factor
    )

    # Scale and align mask the same way
    mask_rgba = np.zeros((*mask_image.shape[:2], 4), dtype=np.uint8)
    if len(mask_image.shape) == 2:
        mask_rgba[:, :, 0] = mask_image
        mask_rgba[:, :, 3] = mask_image
    else:
        mask_rgba[:, :, 0] = mask_image[:, :, 0]
        mask_rgba[:, :, 3] = mask_image[:, :, 0]

    scaled_mask, _ = scale_and_align_image(
        mask_rgba,
        clothed_keypoints,
        base_keypoints,
        config.scale_factor
    )

    # Apply mask to get armor only
    armor = apply_mask_to_image(scaled_clothed, scaled_mask[:, :, 0])

    # TPS warp armor to match base skeleton
    warped_armor = thin_plate_spline_warp(armor, scaled_kpts, base_keypoints)

    # Count uncovered pixels
    uncovered = count_uncovered_pixels(base_image, warped_armor)

    return uncovered, warped_armor
```

**Step 2: Test function compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import evaluate_keypoints; print('OK')"`
Expected: `OK`

---

## Task 7: Add greedy optimizer function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add optimize_keypoints function**

Add after evaluate_keypoints:

```python
@dataclass
class OptimizationResult:
    """Result from skeleton optimization."""
    frame_name: str
    original_keypoints: np.ndarray
    optimized_keypoints: np.ndarray
    initial_uncovered: int
    final_uncovered: int
    iterations: int
    adjustments: Dict[str, Tuple[int, int]]  # keypoint_name -> (dx, dy) total


def optimize_keypoints(
    clothed_image: np.ndarray,
    clothed_keypoints: np.ndarray,
    base_image: np.ndarray,
    base_keypoints: np.ndarray,
    mask_image: np.ndarray,
    config: OptimizerConfig,
    frame_name: str = "",
    verbose: bool = True
) -> OptimizationResult:
    """Optimize clothed keypoints to minimize uncovered pixels.

    Greedy hill-climbing: try 1px moves in each direction for each
    keypoint, keep improvements, stop when no improvement possible.

    Args:
        clothed_image: Original clothed sprite RGBA
        clothed_keypoints: Initial keypoint estimates [14, 2]
        base_image: Target base sprite RGBA
        base_keypoints: Target keypoints [14, 2]
        mask_image: Armor mask
        config: Optimizer configuration
        frame_name: For logging
        verbose: Print progress

    Returns:
        OptimizationResult with optimized keypoints
    """
    original_kpts = clothed_keypoints.copy()
    current_kpts = clothed_keypoints.copy()

    # Initial evaluation
    initial_uncovered, _ = evaluate_keypoints(
        clothed_image, current_kpts, base_image, base_keypoints, mask_image, config
    )
    current_uncovered = initial_uncovered

    if verbose:
        print(f"Initial uncovered pixels: {initial_uncovered}")

    # Directions to try: left, right, up, down
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    iteration = 0
    while current_uncovered > 0 and iteration < config.max_iterations:
        iteration += 1
        improved = False

        for kpt_idx in config.optimize_indices:
            for dx, dy in directions:
                # Try moving this keypoint
                test_kpts = current_kpts.copy()
                test_kpts[kpt_idx, 0] += dx * config.step_size
                test_kpts[kpt_idx, 1] += dy * config.step_size

                test_uncovered, _ = evaluate_keypoints(
                    clothed_image, test_kpts, base_image, base_keypoints,
                    mask_image, config
                )

                if test_uncovered < current_uncovered:
                    current_kpts = test_kpts
                    current_uncovered = test_uncovered
                    improved = True

                    if verbose:
                        print(f"  Iter {iteration}: {KEYPOINT_NAMES[kpt_idx]} "
                              f"({dx:+d},{dy:+d}) -> {current_uncovered} uncovered")

                    if current_uncovered == 0:
                        break

            if current_uncovered == 0:
                break

        if not improved:
            if verbose:
                print(f"  Iter {iteration}: No improvement, stopping")
            break

    # Calculate total adjustments per keypoint
    adjustments = {}
    for i in config.optimize_indices:
        delta = current_kpts[i] - original_kpts[i]
        if np.any(delta != 0):
            adjustments[KEYPOINT_NAMES[i]] = (int(delta[0]), int(delta[1]))

    return OptimizationResult(
        frame_name=frame_name,
        original_keypoints=original_kpts,
        optimized_keypoints=current_kpts,
        initial_uncovered=initial_uncovered,
        final_uncovered=current_uncovered,
        iterations=iteration,
        adjustments=adjustments
    )
```

**Step 2: Test function compiles**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import optimize_keypoints, OptimizationResult; print('OK')"`
Expected: `OK`

---

## Task 8: Add main runner function

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add run_optimization function**

Add at end of file:

```python
def run_optimization(
    frames_dir: Path,
    masks_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    frame_mappings: Optional[Dict[str, str]] = None,
    config: Optional[OptimizerConfig] = None,
    verbose: bool = True
) -> Dict[str, OptimizationResult]:
    """Run optimization on specified frames.

    Args:
        frames_dir: Directory containing frame images
        masks_dir: Directory containing mask images
        annotations_path: Path to annotations.json
        output_dir: Directory to save results
        frame_mappings: Dict of clothed_frame -> base_frame mappings
        config: Optimizer config (uses defaults if None)
        verbose: Print progress

    Returns:
        Dict of frame_name -> OptimizationResult
    """
    if frame_mappings is None:
        frame_mappings = FRAME_MAPPINGS
    if config is None:
        config = OptimizerConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(annotations_path)

    results = {}

    for clothed_name, base_name in frame_mappings.items():
        if verbose:
            print(f"\n=== Optimizing {clothed_name} -> {base_name} ===")

        # Load images
        clothed_img = cv2.imread(str(frames_dir / f"{clothed_name}.png"), cv2.IMREAD_UNCHANGED)
        base_img = cv2.imread(str(frames_dir / f"{base_name}.png"), cv2.IMREAD_UNCHANGED)

        # Get mask (clothed_frame_00 -> mask_00)
        mask_idx = clothed_name.split("_")[-1]
        mask_img = cv2.imread(str(masks_dir / f"mask_{mask_idx}.png"), cv2.IMREAD_UNCHANGED)

        # Get keypoints
        clothed_kpts = get_keypoints_array(annotations, clothed_name)
        base_kpts = get_keypoints_array(annotations, base_name)

        # Run optimization
        result = optimize_keypoints(
            clothed_img, clothed_kpts, base_img, base_kpts,
            mask_img, config, clothed_name, verbose
        )
        results[clothed_name] = result

        if verbose:
            print(f"Final: {result.initial_uncovered} -> {result.final_uncovered} "
                  f"({result.iterations} iterations)")
            if result.adjustments:
                print("Adjustments:")
                for name, (dx, dy) in result.adjustments.items():
                    print(f"  {name}: ({dx:+d}, {dy:+d})")

    # Save results to JSON
    results_dict = {}
    for name, r in results.items():
        results_dict[name] = {
            "original_keypoints": r.original_keypoints.tolist(),
            "optimized_keypoints": r.optimized_keypoints.tolist(),
            "initial_uncovered": r.initial_uncovered,
            "final_uncovered": r.final_uncovered,
            "iterations": r.iterations,
            "adjustments": r.adjustments
        }

    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    if verbose:
        print(f"\nResults saved to {output_dir / 'optimization_results.json'}")

    return results


if __name__ == "__main__":
    # Default paths for CLI usage
    base_dir = Path(__file__).parent.parent / "training_data"

    results = run_optimization(
        frames_dir=base_dir / "frames",
        masks_dir=base_dir / "masks_corrected",
        annotations_path=base_dir / "annotations.json",
        output_dir=base_dir / "skeleton_comparison"
    )
```

**Step 2: Test full module imports**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -c "from sprite_keypoint_detector.optimizer import run_optimization; print('OK')"`
Expected: `OK`

---

## Task 9: Run optimizer on test frames

**Files:**
- None (execution only)

**Step 1: Run the optimizer**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.optimizer
```

Expected:
- Prints initial uncovered pixel counts
- Shows keypoint adjustments as they're found
- Ends with final counts and adjustments summary
- Creates `training_data/skeleton_comparison/optimization_results.json`

**Step 2: Review results**

Run: `cat /Users/roberthyatt/Code/ComfyUI/training_data/skeleton_comparison/optimization_results.json`
Expected: JSON with optimized keypoints for both frames

---

## Task 10: Visualize optimized results

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector/optimizer.py`

**Step 1: Add visualization function**

Add before the `if __name__` block:

```python
def save_comparison_images(
    results: Dict[str, OptimizationResult],
    frames_dir: Path,
    masks_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    config: Optional[OptimizerConfig] = None
) -> None:
    """Save before/after comparison images.

    Args:
        results: Optimization results
        frames_dir: Directory containing frame images
        masks_dir: Directory containing mask images
        annotations_path: Path to annotations.json
        output_dir: Directory to save images
        config: Optimizer config
    """
    if config is None:
        config = OptimizerConfig()

    annotations = load_annotations(annotations_path)

    for clothed_name, result in results.items():
        base_name = FRAME_MAPPINGS[clothed_name]

        # Load images
        clothed_img = cv2.imread(str(frames_dir / f"{clothed_name}.png"), cv2.IMREAD_UNCHANGED)
        base_img = cv2.imread(str(frames_dir / f"{base_name}.png"), cv2.IMREAD_UNCHANGED)
        mask_idx = clothed_name.split("_")[-1]
        mask_img = cv2.imread(str(masks_dir / f"mask_{mask_idx}.png"), cv2.IMREAD_UNCHANGED)
        base_kpts = get_keypoints_array(annotations, base_name)

        # Generate before image (original keypoints)
        _, before_warped = evaluate_keypoints(
            clothed_img, result.original_keypoints, base_img, base_kpts,
            mask_img, config
        )

        # Generate after image (optimized keypoints)
        _, after_warped = evaluate_keypoints(
            clothed_img, result.optimized_keypoints, base_img, base_kpts,
            mask_img, config
        )

        # Composite on base
        def composite(base, overlay):
            result = base.copy()
            mask = overlay[:, :, 3:4] / 255.0
            result[:, :, :3] = (result[:, :, :3] * (1 - mask) + overlay[:, :, :3] * mask).astype(np.uint8)
            result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])
            return result

        before_composite = composite(base_img, before_warped)
        after_composite = composite(base_img, after_warped)

        # Save images
        cv2.imwrite(str(output_dir / f"{clothed_name}_before_opt.png"), before_composite)
        cv2.imwrite(str(output_dir / f"{clothed_name}_after_opt.png"), after_composite)

        print(f"Saved: {clothed_name}_before_opt.png, {clothed_name}_after_opt.png")
```

**Step 2: Update main block to generate images**

Replace the `if __name__` block with:

```python
if __name__ == "__main__":
    # Default paths for CLI usage
    base_dir = Path(__file__).parent.parent / "training_data"

    frames_dir = base_dir / "frames"
    masks_dir = base_dir / "masks_corrected"
    annotations_path = base_dir / "annotations.json"
    output_dir = base_dir / "skeleton_comparison"

    results = run_optimization(
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        annotations_path=annotations_path,
        output_dir=output_dir
    )

    print("\nGenerating comparison images...")
    save_comparison_images(
        results=results,
        frames_dir=frames_dir,
        masks_dir=masks_dir,
        annotations_path=annotations_path,
        output_dir=output_dir
    )
```

**Step 3: Run and generate images**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.optimizer`

Expected: Creates before/after comparison images in skeleton_comparison/

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | Create optimizer.py scaffold | Pending |
| 2 | Add annotation loading | Pending |
| 3 | Add TPS warp function | Pending |
| 4 | Add scale-and-align function | Pending |
| 5 | Add gray pixel counting | Pending |
| 6 | Add pipeline evaluation | Pending |
| 7 | Add greedy optimizer | Pending |
| 8 | Add main runner | Pending |
| 9 | Run optimizer | Pending |
| 10 | Add visualization | Pending |
