#!/usr/bin/env python3
"""Test OpenPose detection thresholds across all frames to find optimal values."""

import json
import sys
from pathlib import Path
from datetime import datetime
from sprite_clothing_gen.comfy_client import ComfyUIClient
import time


def setup_logging():
    """Configure logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/tmp/threshold_test_{timestamp}.log"

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    return log_file


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def setup_test_directories():
    """Create clean output directory structure."""
    output_dir = Path("output/threshold_grid")
    if output_dir.exists():
        log(f"Removing old threshold grid directory...")
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    return output_dir


def get_threshold_grid():
    """Define threshold combinations to test."""
    threshold_1_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    threshold_2_values = [0.01, 0.03, 0.05]

    combinations = []
    for t1 in threshold_1_values:
        for t2 in threshold_2_values:
            combinations.append((t1, t2))

    log(f"Testing {len(combinations)} threshold combinations:")
    log(f"  body_threshold_1: {threshold_1_values}")
    log(f"  body_threshold_2: {threshold_2_values}")
    log(f"  Total tests: {len(combinations)} × 25 frames = {len(combinations) * 25}")

    return combinations


def test_frame_with_thresholds(client: ComfyUIClient, frame_idx: int, thre1: float, thre2: float, output_dir: Path) -> dict:
    """Test OpenPose detection on single frame with specific thresholds.

    Returns:
        dict with success (bool), keypoint_count (int), file_size (int), file_path (str)
    """
    frame_name = f"base_frame_{frame_idx:02d}.png"
    threshold_dir = output_dir / f"thre1_{thre1:.2f}_thre2_{thre2:.2f}"
    threshold_dir.mkdir(exist_ok=True)

    output_prefix = f"threshold_grid/thre1_{thre1:.2f}_thre2_{thre2:.2f}/frame_{frame_idx:02d}"

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
                "body_threshold_1": thre1,
                "body_threshold_2": thre2,
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
            return {
                "success": False,
                "keypoint_count": 0,
                "file_size": 0,
                "file_path": None,
                "error": "Frame not found"
            }

        uploaded = client.upload_image(frame_path)
        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_completion(prompt_id, timeout=30)

        # Find output file
        output_files = list(Path("output").glob(f"{output_prefix}_*.png"))

        if not output_files:
            return {
                "success": False,
                "keypoint_count": 0,
                "file_size": 0,
                "file_path": None,
                "error": "No output file"
            }

        output_file = output_files[0]
        file_size = output_file.stat().st_size

        # Count keypoints (non-black pixels)
        from PIL import Image
        import numpy as np
        img = Image.open(output_file).convert('RGB')
        pixels = np.array(img)
        non_black = np.sum(np.any(pixels > 0, axis=2))

        # Detection threshold: skeleton images are ~4KB+, have keypoints
        success = file_size > 3000 and non_black > 100

        return {
            "success": success,
            "keypoint_count": int(non_black),
            "file_size": int(file_size),
            "file_path": str(output_file),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "keypoint_count": 0,
            "file_size": 0,
            "file_path": None,
            "error": str(e)
        }


def check_disk_space():
    """Warn if output directory is large."""
    output_dir = Path("output")
    if not output_dir.exists():
        return

    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    total_mb = total_size / (1024 * 1024)

    log(f"Current output directory size: {total_mb:.1f} MB")

    if total_mb > 100:
        log("WARNING: Output directory is large (>100MB)")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            sys.exit(1)


if __name__ == "__main__":
    log_file = setup_logging()
    log("OpenPose Threshold Optimization Test Starting")
    log(f"Log file: {log_file}")

    check_disk_space()
    output_dir = setup_test_directories()
    threshold_combinations = get_threshold_grid()

    # Initialize ComfyUI client
    log("Initializing ComfyUI client...")
    client = ComfyUIClient("http://127.0.0.1:8188")
    if not client.health_check():
        log("ERROR: ComfyUI server is not running")
        sys.exit(1)
    log("  ComfyUI server is running")

    # Test single frame to verify function works
    log("\nTesting single frame (0) with default thresholds...")
    result = test_frame_with_thresholds(client, 0, 0.1, 0.05, output_dir)
    log(f"  Result: {result}")
