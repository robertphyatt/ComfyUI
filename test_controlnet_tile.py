#!/usr/bin/env python3
"""Test ControlNet Tile vs OpenPose for pose accuracy."""

import sys
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

# Add ComfyUI to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sprite_clothing_gen.comfy_client import ComfyUIClient


def setup_logging():
    """Configure logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/tmp/tile_test_{timestamp}.log"

    class Logger:
        def __init__(self, log_file):
            self.log_file = log_file
            self.terminal = sys.stdout

        def write(self, message):
            self.terminal.write(message)
            with open(self.log_file, 'a') as f:
                f.write(message)

        def flush(self):
            self.terminal.flush()

    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Logging to {log_file}")
    return log_file


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


if __name__ == "__main__":
    log_file = setup_logging()
    log("ControlNet Tile Test Starting")
    log(f"Working directory: {os.getcwd()}")
