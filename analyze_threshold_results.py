#!/usr/bin/env python3
"""Analyze threshold test results and generate recommendations."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load test results from JSON."""
    results_file = Path("threshold_test_results.json")
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Run test_openpose_thresholds.py first")
        return None

    with open(results_file) as f:
        data = json.load(f)

    print(f"Loaded results from: {results_file}")
    print(f"  Timestamp: {data['timestamp']}")
    print(f"  Total tests: {data['total_tests']}")
    print(f"  Threshold combinations: {data['threshold_combinations']}")

    return data


if __name__ == "__main__":
    print("="*70)
    print("OpenPose Threshold Analysis")
    print("="*70)

    data = load_results()
    if data is None:
        exit(1)
