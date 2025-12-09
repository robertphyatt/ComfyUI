#!/usr/bin/env python3
"""Align clothed frames using OpenPose skeleton keypoint matching."""

import json
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from PIL import Image


def extract_openpose_keypoints(frame_path: str) -> Dict:
    """Extract OpenPose keypoints from a frame using ComfyUI.

    Args:
        frame_path: Path to frame image

    Returns:
        Dict with 'people' array containing pose_keypoints_2d

    Raises:
        RuntimeError: If ComfyUI server not running
    """
    from sprite_clothing_gen.comfy_client import ComfyUIClient
    from sprite_clothing_gen.workflow_builder import build_openpose_preprocessing_workflow

    client = ComfyUIClient("http://127.0.0.1:8188")

    if not client.health_check():
        raise RuntimeError("ComfyUI server not running at http://127.0.0.1:8188")

    # Upload image to ComfyUI
    image_path = Path(frame_path)
    filename = client.upload_image(image_path)

    # Build and queue workflow
    workflow = build_openpose_preprocessing_workflow(filename)
    prompt_id = client.queue_prompt(workflow)

    # Wait for completion
    history = client.wait_for_completion(prompt_id, timeout=60)

    # Extract keypoints from history
    # OpenposePreprocessor outputs keypoints in history['outputs'][node_id]['ui']['openpose_json']
    outputs = history.get('outputs', {})
    for node_id, node_output in outputs.items():
        if 'ui' in node_output and 'openpose_json' in node_output['ui']:
            json_str = node_output['ui']['openpose_json'][0]
            return json.loads(json_str)

    # No keypoints found
    return {'people': []}


def parse_keypoints(keypoints: Dict) -> Tuple[float, float]:
    """Extract upper body center from OpenPose keypoints.

    Uses neck, right shoulder, and left shoulder to calculate center.

    Args:
        keypoints: OpenPose keypoints dict with 'people' array

    Returns:
        (center_x, center_y) tuple

    Raises:
        ValueError: If no people detected
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
