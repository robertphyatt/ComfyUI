#!/usr/bin/env python3
"""Test #22: Optical Flow Warping - Non-AI approach.

Concept: Use optical flow to find how pixels should move from the clothed
reference to match the mannequin pose, then warp the armor accordingly.

This is what you'd do manually - identify corresponding points between
the two images and morph/warp one to match the other.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def load_image(path):
    """Load image as numpy array (BGR for OpenCV)."""
    img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def save_image(arr, path):
    """Save numpy array as image."""
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    img.save(path)

def compute_optical_flow(source, target):
    """Compute dense optical flow from source to target."""
    # Convert to grayscale
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        source_gray, target_gray,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )

    return flow

def warp_image(source, flow):
    """Warp source image using optical flow field."""
    h, w = flow.shape[:2]

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply flow to coordinates
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)

    # Remap (warp) the image
    warped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped

def create_body_mask(image, threshold=245):
    """Create mask where body is (non-white pixels)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < threshold).astype(np.uint8) * 255
    return mask

def blend_with_background(warped, mannequin, mask):
    """Blend warped armor onto mannequin background."""
    # Dilate mask slightly to avoid edge artifacts
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # Normalize mask to 0-1
    mask_norm = mask_dilated.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)

    # Blend: warped armor where mask is white, mannequin background where black
    result = (warped * mask_3ch + mannequin * (1 - mask_3ch)).astype(np.uint8)

    return result

def process_frame(clothed_path, mannequin_path, output_path):
    """Warp clothed frame to match mannequin pose."""
    print(f"Processing: {clothed_path.name} -> {mannequin_path.name}")

    # Load images
    clothed = load_image(clothed_path)
    mannequin = load_image(mannequin_path)

    # Compute optical flow from clothed to mannequin
    # This tells us how pixels need to move
    flow = compute_optical_flow(clothed, mannequin)

    # Warp the clothed image to match mannequin pose
    warped = warp_image(clothed, flow)

    # Create mask from mannequin (where the body is)
    mask = create_body_mask(mannequin)

    # Blend warped armor with clean background
    result = blend_with_background(warped, mannequin, mask)

    # Save result
    save_image(result, output_path)
    print(f"  Saved: {output_path}")

    # Also save intermediate results for debugging
    debug_dir = output_path.parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Visualize flow
    flow_vis = visualize_flow(flow)
    save_image(flow_vis, debug_dir / f"flow_{output_path.stem}.png")

    # Save warped before blending
    save_image(warped, debug_dir / f"warped_{output_path.stem}.png")

    # Save mask
    cv2.imwrite(str(debug_dir / f"mask_{output_path.stem}.png"), mask)

def visualize_flow(flow):
    """Visualize optical flow as HSV image."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def run_test():
    print("=" * 70)
    print("TEST #22: Optical Flow Warping (Non-AI)")
    print("=" * 70)
    print()
    print("Approach: Compute how pixels move from clothed->mannequin, warp armor")
    print()

    input_dir = Path("/Users/roberthyatt/Code/ComfyUI/input")
    output_dir = Path("/Users/roberthyatt/Code/ComfyUI/output/test22_optical_flow")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [1, 5, 10, 15]

    for frame_num in frames:
        clothed_path = input_dir / f"clothed_frame_{frame_num:02d}.png"
        mannequin_path = input_dir / f"base_frame_{frame_num:02d}.png"
        output_path = output_dir / f"frame_{frame_num:02d}.png"

        if clothed_path.exists() and mannequin_path.exists():
            process_frame(clothed_path, mannequin_path, output_path)
        else:
            print(f"Skipping frame {frame_num} - files not found")

    print(f"\nDone! Check {output_dir}")

if __name__ == "__main__":
    run_test()
