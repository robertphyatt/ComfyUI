#!/usr/bin/env python3
"""Simple test to see where SAM is failing."""

import sys
import time
sys.path.insert(0, "custom_nodes/comfyui_controlnet_aux/src")

print("Step 1: Importing torch...")
import torch
print(f"  ✓ Torch imported")

print("\nStep 2: Checking device...")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"  ✓ Device: {device}")

print("\nStep 3: Importing SamDetector...")
from custom_controlnet_aux.sam import SamDetector
print(f"  ✓ SamDetector imported")

print("\nStep 4: Creating SamDetector instance...")
start_time = time.time()
try:
    sam = SamDetector.from_pretrained()
    elapsed = time.time() - start_time
    print(f"  ✓ SamDetector created in {elapsed:.2f}s")

    print("\nStep 5: Moving to device...")
    start_time = time.time()
    sam = sam.to(device)
    elapsed = time.time() - start_time
    print(f"  ✓ Moved to {device} in {elapsed:.2f}s")

    print("\n✓ All steps completed successfully!")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
