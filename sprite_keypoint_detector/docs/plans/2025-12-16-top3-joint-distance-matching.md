# Top 3 Joint Distance Matching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change frame matching to select from top 3 candidates by joint distance instead of top 5, ensuring pose similarity takes precedence over red pixel count.

**Architecture:** Modify `find_top_candidates()` default parameter and update `select_best_match()` call in pipeline to use top 3 instead of top 5. This narrows the candidate pool to better pose matches before applying red-pixel scoring.

**Tech Stack:** Python, existing matching.py and pipeline.py modules

---

### Task 1: Update find_top_candidates default parameter

**Files:**
- Modify: `sprite_keypoint_detector/matching.py:66-78`

**Step 1: Change top_n default from 5 to 3**

In `matching.py`, find `find_top_candidates` function and change the default:

```python
def find_top_candidates(
    base_frame: str,
    base_keypoints: Dict,
    clothed_annotations: Dict[str, Dict],
    top_n: int = 3  # Changed from 5 to 3
) -> List[Tuple[str, float]]:
```

**Step 2: Run pipeline to verify change works**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --frames-dir training_data/frames/ --annotations training_data/annotations.json --masks training_data/masks_corrected/ --output /tmp/pipeline_test/ --skip-validation 2>&1 | grep -A 5 "Matching base_frame_21"`

Expected: Should show "Top 3 by joint distance" with only 3 candidates listed

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add matching.py
git commit -m "fix(matching): narrow candidates to top 3 by joint distance

Reduces candidate pool from top 5 to top 3 by joint distance before
applying red-pixel scoring. This ensures pose similarity takes
precedence - a slightly higher red count with much better pose match
will now be selected over a poor pose match with lower red.

Fixes issue where clothed_frame_08 (wrong pose) was selected over
clothed_frame_09/21 (correct pose) for base_frame_21.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Update pipeline print statement to reflect top 3

**Files:**
- Modify: `sprite_keypoint_detector/pipeline.py:255`

**Step 1: Update debug print to say "Top 3" instead of "Top 5"**

Find line with `print(f"  Top 5 by joint distance:` and change to:

```python
print(f"  Top 3 by joint distance: {[c[0] for c in candidates]}")
```

**Step 2: Verify output**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --frames-dir training_data/frames/ --annotations training_data/annotations.json --masks training_data/masks_corrected/ --output /tmp/pipeline_test/ --skip-validation 2>&1 | grep "Top 3"`

Expected: Lines showing "Top 3 by joint distance: [...]"

**Step 3: Commit**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git add pipeline.py
git commit -m "chore(pipeline): update debug output to say Top 3

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Verify frame 21 now matches correctly

**Files:**
- None (verification only)

**Step 1: Run full pipeline with debug**

Run: `cd /Users/roberthyatt/Code/ComfyUI && python3 -m sprite_keypoint_detector.pipeline --frames-dir training_data/frames/ --annotations training_data/annotations.json --masks training_data/masks_corrected/ --output /tmp/pipeline_test/ --skip-validation --debug 2>&1 | grep -A 6 "Matching base_frame_21"`

Expected output should show:
- Top 3 candidates: clothed_frame_09, clothed_frame_21, clothed_frame_10
- Best match: clothed_frame_09.png (red=1958) [OK]

**Step 2: Visually verify frame 21 comparison**

Open: `/tmp/pipeline_test/debug/comparison/frame_21.png`

Expected: Legs should look correct without the distortion seen before

**Step 3: Push changes**

```bash
cd /Users/roberthyatt/Code/ComfyUI/sprite_keypoint_detector
git push
```

---

## Summary

This is a minimal 2-line change:
1. `matching.py`: Change `top_n: int = 5` to `top_n: int = 3`
2. `pipeline.py`: Change `"Top 5"` to `"Top 3"` in print statement

The logic already works correctly - we just need to narrow the candidate pool so bad pose matches can't win purely on red pixel count.
