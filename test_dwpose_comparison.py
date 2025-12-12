#!/usr/bin/env python3
"""Compare DWPose vs OpenPose on frames that OpenPose fails to detect."""

import json
import sys
from pathlib import Path
from datetime import datetime
from sprite_clothing_gen.comfy_client import ComfyUIClient
from PIL import Image
import numpy as np

# Frames that OpenPose fails on (0 pixels detected at any threshold)
FAILING_FRAMES = [0, 2, 6, 9, 12, 14, 18, 22]

# Also test a few succeeding frames as control
CONTROL_FRAMES = [1, 3, 4, 8]


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def test_frame_dwpose(client: ComfyUIClient, frame_idx: int, output_dir: Path) -> dict:
    """Test DWPose detection on a single frame."""
    frame_name = f"base_frame_{frame_idx:02d}.png"
    output_prefix = f"dwpose_test/frame_{frame_idx:02d}"

    workflow = {
        "1": {
            "inputs": {"image": frame_name},
            "class_type": "LoadImage"
        },
        "2": {
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384.onnx",
                "image": ["1", 0]
            },
            "class_type": "DWPreprocessor"
        },
        "3": {
            "inputs": {
                "filename_prefix": output_prefix,
                "images": ["2", 0]
            },
            "class_type": "SaveImage"
        }
    }

    try:
        frame_path = Path("training_data/frames") / frame_name
        if not frame_path.exists():
            return {"success": False, "keypoint_count": 0, "error": "Frame not found"}

        client.upload_image(frame_path)
        prompt_id = client.queue_prompt(workflow)
        client.wait_for_completion(prompt_id, timeout=60)

        # Find output file
        output_files = list(Path("output").glob(f"{output_prefix}_*.png"))
        if not output_files:
            return {"success": False, "keypoint_count": 0, "error": "No output file"}

        output_file = output_files[0]

        # Count non-black pixels
        img = Image.open(output_file).convert('RGB')
        pixels = np.array(img)
        non_black = int(np.sum(np.any(pixels > 0, axis=2)))

        success = non_black > 100
        return {
            "success": success,
            "keypoint_count": non_black,
            "file_path": str(output_file),
            "error": None
        }

    except Exception as e:
        return {"success": False, "keypoint_count": 0, "error": str(e)}


def test_frame_openpose(client: ComfyUIClient, frame_idx: int, output_dir: Path) -> dict:
    """Test OpenPose detection on a single frame (for comparison)."""
    frame_name = f"base_frame_{frame_idx:02d}.png"
    output_prefix = f"openpose_test/frame_{frame_idx:02d}"

    workflow = {
        "1": {
            "inputs": {"image": frame_name},
            "class_type": "LoadImage"
        },
        "2": {
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 512,
                "body_threshold_1": 0.01,
                "body_threshold_2": 0.01,
                "image": ["1", 0]
            },
            "class_type": "OpenposePreprocessor"
        },
        "3": {
            "inputs": {
                "filename_prefix": output_prefix,
                "images": ["2", 0]
            },
            "class_type": "SaveImage"
        }
    }

    try:
        frame_path = Path("training_data/frames") / frame_name
        if not frame_path.exists():
            return {"success": False, "keypoint_count": 0, "error": "Frame not found"}

        client.upload_image(frame_path)
        prompt_id = client.queue_prompt(workflow)
        client.wait_for_completion(prompt_id, timeout=60)

        output_files = list(Path("output").glob(f"{output_prefix}_*.png"))
        if not output_files:
            return {"success": False, "keypoint_count": 0, "error": "No output file"}

        output_file = output_files[0]

        img = Image.open(output_file).convert('RGB')
        pixels = np.array(img)
        non_black = int(np.sum(np.any(pixels > 0, axis=2)))

        success = non_black > 100
        return {
            "success": success,
            "keypoint_count": non_black,
            "file_path": str(output_file),
            "error": None
        }

    except Exception as e:
        return {"success": False, "keypoint_count": 0, "error": str(e)}


if __name__ == "__main__":
    log("DWPose vs OpenPose Comparison Test")
    log("="*60)

    # Setup output directories
    output_dir = Path("output")
    (output_dir / "dwpose_test").mkdir(parents=True, exist_ok=True)
    (output_dir / "openpose_test").mkdir(parents=True, exist_ok=True)

    # Initialize client
    log("Connecting to ComfyUI...")
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        log("ERROR: ComfyUI server not running")
        sys.exit(1)
    log("Connected.")

    # Test frames
    all_frames = FAILING_FRAMES + CONTROL_FRAMES
    results = {"dwpose": {}, "openpose": {}}

    log(f"\nTesting {len(all_frames)} frames...")
    log(f"OpenPose-failing frames: {FAILING_FRAMES}")
    log(f"Control frames: {CONTROL_FRAMES}")

    for frame_idx in all_frames:
        log(f"\n--- Frame {frame_idx:02d} ---")

        # Test DWPose
        log(f"  DWPose...")
        dw_result = test_frame_dwpose(client, frame_idx, output_dir)
        results["dwpose"][frame_idx] = dw_result
        dw_status = "DETECTED" if dw_result["success"] else "FAILED"
        log(f"    {dw_status} (pixels: {dw_result['keypoint_count']})")

        # Test OpenPose
        log(f"  OpenPose...")
        op_result = test_frame_openpose(client, frame_idx, output_dir)
        results["openpose"][frame_idx] = op_result
        op_status = "DETECTED" if op_result["success"] else "FAILED"
        log(f"    {op_status} (pixels: {op_result['keypoint_count']})")

    # Summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)

    log(f"\n{'Frame':<8} {'OpenPose':<12} {'DWPose':<12} {'Winner'}")
    log("-"*44)

    dwpose_wins = 0
    openpose_wins = 0
    ties = 0

    for frame_idx in all_frames:
        op_ok = results["openpose"][frame_idx]["success"]
        dw_ok = results["dwpose"][frame_idx]["success"]

        op_str = "DETECTED" if op_ok else "FAILED"
        dw_str = "DETECTED" if dw_ok else "FAILED"

        if dw_ok and not op_ok:
            winner = "DWPose"
            dwpose_wins += 1
        elif op_ok and not dw_ok:
            winner = "OpenPose"
            openpose_wins += 1
        elif dw_ok and op_ok:
            winner = "TIE"
            ties += 1
        else:
            winner = "BOTH FAIL"

        marker = " <-- OpenPose failure" if frame_idx in FAILING_FRAMES else ""
        log(f"{frame_idx:02d}       {op_str:<12} {dw_str:<12} {winner}{marker}")

    log(f"\nOn OpenPose-failing frames ({len(FAILING_FRAMES)} frames):")
    failing_dw_success = sum(1 for f in FAILING_FRAMES if results["dwpose"][f]["success"])
    log(f"  DWPose detected: {failing_dw_success}/{len(FAILING_FRAMES)}")

    log(f"\nOverall:")
    log(f"  DWPose wins: {dwpose_wins}")
    log(f"  OpenPose wins: {openpose_wins}")
    log(f"  Ties: {ties}")

    # Save results
    results_file = Path("dwpose_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {results_file}")
