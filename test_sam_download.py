#!/usr/bin/env python3
"""Test SAM model download with progress."""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'debug'

from transformers import SamModel, SamProcessor

print("Downloading SAM model with verbose logging...")
print("This may take a few minutes for the 346MB model file...\n")

try:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print("✓ Processor loaded")

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    print("✓ Model loaded")

    print("\n✓ SUCCESS: Model fully downloaded and loaded")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
