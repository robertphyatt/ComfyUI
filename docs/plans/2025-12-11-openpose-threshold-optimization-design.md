# OpenPose Threshold Optimization Design

**Date:** 2025-12-11
**Goal:** Find optimal OpenPose detection thresholds to achieve 100% detection success on all 25 sprite frames
**Current State:** 11/25 frames detect with default thresholds (0.1, 0.05); behavior is 100% deterministic

---

## Problem Statement

OpenPose detects poses on 11 frames but fails on 14 frames. The detection thresholds `body_threshold_1` (default 0.1) and `body_threshold_2` (default 0.05) control keypoint sensitivity. We exposed these as configurable parameters. Now we must find threshold values that detect all 25 frames while avoiding false positives.

## Testing Strategy

### Grid Search Approach

Test 18 threshold combinations across all 25 frames (450 tests total, ~15 minutes):

**Thresholds to test:**
- `body_threshold_1`: [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
- `body_threshold_2`: [0.01, 0.03, 0.05]

This coarse grid reveals the detection landscape quickly. If no single threshold pair works for all frames, we run a finer-grained search in the promising region.

### Data Collection

For each test, capture:
1. Success/failure (file size > 3KB indicates detection)
2. Keypoint count (non-black pixels in skeleton output)
3. Output image path for visual inspection

Keypoint counts distinguish quality detections from false positives. Many keypoints signal good pose detection; few keypoints suggest garbage.

## Implementation Components

### Test Script: `test_openpose_thresholds.py`

Executes the grid search:

1. Clean `output/threshold_grid/` directory
2. For each threshold combination:
   - Test all 25 frames (one run per frame, determinism proven)
   - Save skeleton output to `output/threshold_grid/thre1_{value}_thre2_{value}/frame_{idx:02d}.png`
   - Record results to JSON
3. Save complete results to `threshold_test_results.json`
4. Log progress to console and `/tmp/threshold_test.log`

**Error Handling:**
- Check ComfyUI server health before starting
- Save partial results if interrupted
- Warn if output directory exceeds 100MB

### Analysis Script: `analyze_threshold_results.py`

Generates insights from test results:

**1. Detection Success Heatmap**
- Axes: `body_threshold_1` (X) vs. `body_threshold_2` (Y)
- Cell value: Number of frames detected (0-25)
- Output: `threshold_heatmap.png`

**2. Per-Frame Sensitivity Report**
- For each failing frame, show threshold where detection starts
- Plot keypoint count vs. threshold curves
- Identify frames requiring lowest thresholds

**3. Keypoint Quality Analysis**
- Compare keypoint counts across thresholds
- Flag suspected false positives (few keypoints)
- Find sweet spot: maximum detection with good quality

**4. Recommendations**
- Single threshold pair working for all 25 frames (if exists)
- If none works: minimum threshold per frame
- Trade-off analysis: coverage vs. false positive risk

**Outputs:**
- `threshold_heatmap.png` - Visual detection grid
- `frame_sensitivity_report.txt` - Per-frame requirements
- `recommended_thresholds.json` - Final values

## Integration

After finding optimal thresholds:

1. Update `sprite_clothing_gen/workflow_builder.py` to use optimal values
2. Pass thresholds to OpenposePreprocessor node in workflow
3. Re-run full pipeline on all 25 frames
4. Visually inspect skeleton outputs for false positives
5. Verify detected poses match mannequin positions

## Success Criteria

- All 25 frames detect poses successfully
- Detected skeletons are usable (sufficient keypoints)
- No false positives (garbage detections)
- Skeleton poses match mannequin body positions

## Deliverables

1. `test_openpose_thresholds.py` - Grid search implementation
2. `analyze_threshold_results.py` - Analysis and visualization
3. `threshold_test_results.json` - Complete test data
4. `threshold_heatmap.png` - Visual results
5. `recommended_thresholds.json` - Optimal values
6. `docs/findings/2025-12-11-openpose-threshold-optimization.md` - Summary of findings and recommendations
