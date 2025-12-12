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


if __name__ == "__main__":
    log_file = setup_logging()
    log("OpenPose Threshold Optimization Test Starting")
    log(f"Log file: {log_file}")
