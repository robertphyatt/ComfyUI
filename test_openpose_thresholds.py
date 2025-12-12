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
