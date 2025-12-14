# Fix Pipeline Crashes and Verify OpenPose Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix pipeline crash caused by missing return codes, investigate why all alignment offsets are zero, and verify previous coordinate conversion fixes actually work.

**Architecture:** Debug the OpenPose keypoint extraction to verify normalized coordinates are being converted to pixels correctly. Fix return code handling in all pipeline scripts. Add comprehensive logging to trace the alignment calculation. Verify fixes with code reviews and end-to-end tests.

**Tech Stack:** Python, OpenPose via ComfyUI, numpy, PIL

---

## Issues to Fix

1. **Pipeline crash**: `extend_armor_feet.py` main() returns None instead of 0, causing `if result != 0:` to evaluate True
2. **Zero offsets**: All frames show `(+0, +0)` offset despite coordinate conversion "fix"
3. **Verification needed**: Code review of keypoint extraction and coordinate conversion fixes

---

### Task 1: Fix Return Codes in All Pipeline Scripts

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/extend_armor_feet.py:50-87`
- Modify: `/Users/roberthyatt/Code/ComfyUI/extract_clothing_final.py` (if needed)
- Modify: `/Users/roberthyatt/Code/ComfyUI/validate_predicted_masks.py` (if needed)

**Step 1: Check all main() functions return proper exit codes**

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI
grep -n "def main()" extend_armor_feet.py extract_clothing_final.py validate_predicted_masks.py
grep -A 50 "def main()" extend_armor_feet.py | grep "return"
```

Expected: Identify which functions don't return 0 on success

**Step 2: Fix extend_armor_feet.py to return 0**

Add at end of main() function (before the closing of the function):

```python
    print("=" * 70)
    print("✓ Armor extended for all frames")
    print("=" * 70)

    return 0  # ADD THIS LINE


if __name__ == "__main__":
    main()
```

**Step 3: Fix extract_clothing_final.py to return 0**

Add at end of main() function:

```python
    print()
    print("=" * 70)
    print(f"✓ Final clothing spritesheet saved to {output_path}")
    print("=" * 70)

    return 0  # ADD THIS LINE


if __name__ == "__main__":
    main()
```

**Step 4: Verify validate_predicted_masks.py returns properly**

Check if it has return statement. If not, add `return 0` at end of main().

**Step 5: Test the pipeline gets past armor extension**

Run:
```bash
# This will fail at later step, but should get past armor extension
python process_clothing_spritesheet.py
```

Expected: Should complete Step 2 without "ERROR: Armor extension failed"

**Step 6: Commit return code fixes**

```bash
git add extend_armor_feet.py extract_clothing_final.py validate_predicted_masks.py
git commit -m "fix: ensure all pipeline scripts return proper exit codes

All main() functions must return 0 on success, not None.
Pipeline checks 'if result != 0' which evaluates True for None,
causing false failures."
```

---

### Task 2: Add Debugging to Investigate Zero Offsets

**Files:**
- Modify: `/Users/roberthyatt/Code/ComfyUI/align_with_openpose.py:92-117`

**Step 1: Add detailed logging to calculate_alignment_offset()**

Replace the function with enhanced logging version:

```python
def calculate_alignment_offset(base_keypoints: Dict, clothed_keypoints: Dict, image_size: int = 512) -> Tuple[int, int]:
    """Calculate pixel offset needed to align clothed frame to base frame.

    Args:
        base_keypoints: OpenPose keypoints for base frame
        clothed_keypoints: OpenPose keypoints for clothed frame
        image_size: Image dimensions in pixels (default 512)

    Returns:
        (offset_x, offset_y) tuple in pixels
    """
    # Get normalized centers (0.0 to 1.0)
    base_center_x, base_center_y = parse_keypoints(base_keypoints)
    clothed_center_x, clothed_center_y = parse_keypoints(clothed_keypoints)

    # DEBUG: Print normalized coordinates
    print(f"    DEBUG: Base center (normalized): ({base_center_x:.4f}, {base_center_y:.4f})")
    print(f"    DEBUG: Clothed center (normalized): ({clothed_center_x:.4f}, {clothed_center_y:.4f})")

    # Convert to pixel coordinates BEFORE calculating offset
    base_px_x = base_center_x * image_size
    base_px_y = base_center_y * image_size
    clothed_px_x = clothed_center_x * image_size
    clothed_px_y = clothed_center_y * image_size

    # DEBUG: Print pixel coordinates
    print(f"    DEBUG: Base center (pixels): ({base_px_x:.2f}, {base_px_y:.2f})")
    print(f"    DEBUG: Clothed center (pixels): ({clothed_px_x:.2f}, {clothed_px_y:.2f})")

    # Calculate pixel offset
    offset_x = int(base_px_x - clothed_px_x)
    offset_y = int(base_px_y - clothed_px_y)

    # DEBUG: Print offset calculation
    print(f"    DEBUG: Raw offset: ({base_px_x - clothed_px_x:.2f}, {base_px_y - clothed_px_y:.2f})")
    print(f"    DEBUG: Final offset (int): ({offset_x:+d}, {offset_y:+d})")

    return offset_x, offset_y
```

**Step 2: Run pipeline on frame 00 only to see debug output**

Temporarily modify align_with_openpose.py main() to process only frame 0:

```python
for frame_idx in range(1):  # Change from range(25) to range(1)
```

Run:
```bash
python align_with_openpose.py 2>&1 | tee debug_alignment.log
```

Expected: See detailed coordinate values showing where the zero offset comes from

**Step 3: Analyze the debug output**

Check debug_alignment.log for:
- Are normalized coordinates identical? (e.g., both 0.4921, 0.4921)
- Are pixel coordinates identical after multiplication?
- Is the raw offset near zero (e.g., -0.3 pixels)?

**Step 4: Save findings for next task**

Create `/tmp/alignment_debug_findings.txt` with observations:
- What are the actual coordinate values?
- Are base and clothed frames already aligned in the source data?
- Is OpenPose detecting the same position in both frames?

---

### Task 3: Code Review - Verify Keypoint Extraction Fix

**Files:**
- Review: `/Users/roberthyatt/Code/ComfyUI/align_with_openpose.py:11-56`
- Compare: `/Users/roberthyatt/Code/ComfyUI/debug_openpose_history.json` (if exists)

**Step 1: Verify the history structure matches code**

Check the actual history JSON from debug run:

```bash
cd /Users/roberthyatt/Code/ComfyUI
# If debug file exists from previous run
if [ -f debug_openpose_history.json ]; then
    jq '.outputs | keys' debug_openpose_history.json
    jq '.outputs."2" | keys' debug_openpose_history.json
fi
```

Expected: Should show `["openpose_json"]` at `outputs."2"`, NOT under a `ui` key

**Step 2: Manually verify keypoint extraction on one frame**

Create test script `/tmp/test_keypoint_extraction.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/roberthyatt/Code/ComfyUI')

from align_with_openpose import extract_openpose_keypoints
import json

# Extract from one frame
keypoints = extract_openpose_keypoints('training_data/frames/base_frame_00.png')

print("=== EXTRACTED KEYPOINTS ===")
print(json.dumps(keypoints, indent=2)[:500])
print("\n=== PEOPLE DETECTED ===")
print(f"Number of people: {len(keypoints.get('people', []))}")

if keypoints.get('people'):
    person = keypoints['people'][0]
    pose_kp = person.get('pose_keypoints_2d', [])
    print(f"Number of keypoint values: {len(pose_kp)}")
    print(f"First 15 values: {pose_kp[:15]}")

    # Check if normalized
    max_val = max(pose_kp)
    min_val = min([v for v in pose_kp if v > 0])
    print(f"Max value: {max_val}, Min value: {min_val}")
    if max_val <= 1.0:
        print("✓ Coordinates appear to be NORMALIZED (0-1)")
    else:
        print("✗ Coordinates appear to be ABSOLUTE pixels")
```

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
python /tmp/test_keypoint_extraction.py
```

Expected:
- Should extract keypoints successfully
- Should show normalized coordinates (0.0 to 1.0)
- Should detect 1 person with 54 keypoint values

**Step 3: Document findings**

Create summary in `/tmp/keypoint_extraction_review.txt`:
- Does extraction work? Yes/No
- Are coordinates normalized? Yes/No
- Any issues found?

---

### Task 4: Code Review - Verify Coordinate Conversion Fix

**Files:**
- Review: `/Users/roberthyatt/Code/ComfyUI/align_with_openpose.py:92-117`
- Review: `/Users/roberthyatt/Code/ComfyUI/tests/test_openpose_alignment.py:62-100`

**Step 1: Verify the test actually tests normalized coordinates**

Check the test values:

```python
# From test file - are these normalized?
0.2, 0.2, 0.9,  # neck
0.18, 0.22, 0.9,  # right shoulder
0.22, 0.22, 0.9,  # left shoulder
```

These ARE normalized (0.0-1.0 range). Test is correct.

**Step 2: Manually calculate expected offset from test**

Using test data:
- Base center: (0.2 + 0.18 + 0.22) / 3 = 0.2, (0.2 + 0.22 + 0.22) / 3 = 0.2133
- Clothed center: (0.22 + 0.20 + 0.24) / 3 = 0.22, (0.21 + 0.23 + 0.23) / 3 = 0.2233
- In pixels: Base = (102.4, 109.2), Clothed = (112.6, 114.3)
- Offset: (102.4 - 112.6, 109.2 - 114.3) = (-10.2, -5.1) → int = (-10, -5)

Expected: Test should pass with offset (-10, -5)

**Step 3: Run the test to verify it passes**

```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
pytest tests/test_openpose_alignment.py::test_calculate_alignment_offset -v
```

Expected: PASS

**Step 4: If test passes but real data shows zero offset**

This means:
- The LOGIC is correct (test proves it)
- The REAL DATA has identical positions in base/clothed frames

Hypothesis: The source frames might already be aligned, OR OpenPose is detecting the same skeleton position because the character is in the same place in both images.

---

### Task 5: Investigate Why Real Frames Produce Zero Offset

**Files:**
- Investigate: `/Users/roberthyatt/Code/ComfyUI/training_data/frames/base_frame_00.png`
- Investigate: `/Users/roberthyatt/Code/ComfyUI/training_data/frames/clothed_frame_00.png`

**Step 1: Visually inspect the frames**

Create visualization script `/tmp/compare_frames.py`:

```python
#!/usr/bin/env python3
from PIL import Image, ImageDraw
import numpy as np

# Load frames
base = Image.open('/Users/roberthyatt/Code/ComfyUI/training_data/frames/base_frame_00.png').convert('RGBA')
clothed = Image.open('/Users/roberthyatt/Code/ComfyUI/training_data/frames/clothed_frame_00.png').convert('RGBA')

# Create side-by-side comparison
width, height = base.size
comparison = Image.new('RGBA', (width * 2, height))
comparison.paste(base, (0, 0))
comparison.paste(clothed, (width, 0))

# Save
comparison.save('/tmp/frame_comparison.png')
print("Saved comparison to /tmp/frame_comparison.png")

# Calculate center of mass for each
def get_center_of_mass(img):
    arr = np.array(img)
    alpha = arr[:, :, 3]
    y_indices, x_indices = np.where(alpha > 128)
    if len(x_indices) == 0:
        return None, None
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)
    return center_x, center_y

base_cx, base_cy = get_center_of_mass(base)
clothed_cx, clothed_cy = get_center_of_mass(clothed)

print(f"\nCenter of mass (bounding box method):")
print(f"Base: ({base_cx:.1f}, {base_cy:.1f})")
print(f"Clothed: ({clothed_cx:.1f}, {clothed_cy:.1f})")
print(f"Difference: ({base_cx - clothed_cx:.1f}, {base_cy - clothed_cy:.1f})")
```

Run:
```bash
python /tmp/compare_frames.py
```

Expected: Shows whether characters are actually in different positions

**Step 2: Extract and compare actual OpenPose keypoints**

Create comparison script `/tmp/compare_keypoints.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/roberthyatt/Code/ComfyUI')

from align_with_openpose import extract_openpose_keypoints, parse_keypoints

# Extract from both frames
print("Extracting base frame keypoints...")
base_kp = extract_openpose_keypoints('training_data/frames/base_frame_00.png')

print("Extracting clothed frame keypoints...")
clothed_kp = extract_openpose_keypoints('training_data/frames/clothed_frame_00.png')

# Parse centers
base_cx, base_cy = parse_keypoints(base_kp)
clothed_cx, clothed_cy = parse_keypoints(clothed_kp)

print("\n=== COMPARISON ===")
print(f"Base center (normalized): ({base_cx:.6f}, {base_cy:.6f})")
print(f"Clothed center (normalized): ({clothed_cx:.6f}, {clothed_cy:.6f})")
print(f"Difference (normalized): ({abs(base_cx - clothed_cx):.6f}, {abs(base_cy - clothed_cy):.6f})")

print(f"\nBase center (pixels @ 512): ({base_cx * 512:.2f}, {base_cy * 512:.2f})")
print(f"Clothed center (pixels @ 512): ({clothed_cx * 512:.2f}, {clothed_cy * 512:.2f})")
print(f"Difference (pixels): ({abs(base_cx - clothed_cx) * 512:.2f}, {abs(base_cy - clothed_cy) * 512:.2f})")

offset_x = int((base_cx - clothed_cx) * 512)
offset_y = int((base_cy - clothed_cy) * 512)
print(f"\nCalculated offset: ({offset_x:+d}, {offset_y:+d})")

# Get individual keypoints
if base_kp.get('people') and clothed_kp.get('people'):
    base_pose = base_kp['people'][0]['pose_keypoints_2d']
    clothed_pose = clothed_kp['people'][0]['pose_keypoints_2d']

    print("\n=== NECK POSITIONS (index 1) ===")
    print(f"Base neck: ({base_pose[3]:.4f}, {base_pose[4]:.4f})")
    print(f"Clothed neck: ({clothed_pose[3]:.4f}, {clothed_pose[4]:.4f})")

    print("\n=== RIGHT SHOULDER (index 2) ===")
    print(f"Base R-shoulder: ({base_pose[6]:.4f}, {base_pose[7]:.4f})")
    print(f"Clothed R-shoulder: ({clothed_pose[6]:.4f}, {clothed_pose[7]:.4f})")

    print("\n=== LEFT SHOULDER (index 5) ===")
    print(f"Base L-shoulder: ({base_pose[15]:.4f}, {base_pose[16]:.4f})")
    print(f"Clothed L-shoulder: ({clothed_pose[15]:.4f}, {clothed_pose[16]:.4f})")
```

Run:
```bash
cd /Users/roberthyatt/Code/ComfyUI
source .venv/bin/activate
python /tmp/compare_keypoints.py
```

Expected: Shows the actual keypoint coordinates and calculated offset

**Step 3: Determine root cause**

Based on output, determine:

**If coordinates are nearly identical (< 0.01 difference):**
- Root cause: Source frames are already aligned
- Solution: No alignment needed, or frames were pre-aligned

**If coordinates differ significantly (> 0.05 difference) but offset is still zero:**
- Root cause: Bug in coordinate handling
- Solution: Need to trace through calculation step-by-step

**If coordinates differ slightly (0.01-0.05) producing sub-pixel offsets:**
- Root cause: Differences are sub-pixel, int() rounds to zero
- Solution: Use round() instead of int(), or accept sub-pixel precision

---

### Task 6: Implement Fix Based on Root Cause

**Scenario A: Frames Are Already Aligned (No Fix Needed)**

If investigation shows frames are identical position:

**Step 1: Document finding**

Add note to plan and commit message explaining frames are pre-aligned.

**Step 2: Consider skipping alignment step**

Add check at start of align_with_openpose.py main():

```python
print("Checking if frames need alignment...")
# Sample first frame
base_kp = extract_openpose_keypoints(str(frames_dir / "base_frame_00.png"))
clothed_kp = extract_openpose_keypoints(str(frames_dir / "clothed_frame_00.png"))
offset_x, offset_y = calculate_alignment_offset(base_kp, clothed_kp)

if offset_x == 0 and offset_y == 0:
    print("✓ Frames are already aligned, copying directly...")
    for frame_idx in range(25):
        src = frames_dir / f"clothed_frame_{frame_idx:02d}.png"
        dst = output_dir / f"clothed_frame_{frame_idx:02d}.png"
        shutil.copy(src, dst)
    return 0
```

---

**Scenario B: Sub-Pixel Rounding Issue**

If offsets are 0.3-0.9 pixels (getting rounded to zero):

**Step 1: Use round() instead of int()**

Modify calculate_alignment_offset():

```python
# Calculate pixel offset with rounding instead of truncation
offset_x = round(base_px_x - clothed_px_x)
offset_y = round(base_px_y - clothed_px_y)
```

**Step 2: Update test to use round()**

Update test expectations to match round() behavior.

**Step 3: Test with real data**

```bash
python align_with_openpose.py
```

Expected: Should see non-zero offsets (e.g., +1, -1)

---

**Scenario C: Actual Bug in Calculation**

If coordinates are different but offset is still zero:

**Step 1: Add assert statements to trace bug**

```python
# After getting normalized centers
assert 0.0 <= base_center_x <= 1.0, f"Base X out of range: {base_center_x}"
assert 0.0 <= base_center_y <= 1.0, f"Base Y out of range: {base_center_y}"

# After pixel conversion
assert base_px_x >= 0 and base_px_x <= image_size, f"Base pixel X out of range: {base_px_x}"

# Before return
assert not (offset_x == 0 and offset_y == 0 and base_px_x != clothed_px_x), \
    f"Offset is zero but pixels differ: base=({base_px_x}, {base_px_y}), clothed=({clothed_px_x}, {clothed_px_y})"
```

**Step 2: Run with assertions to find where logic breaks**

```bash
python align_with_openpose.py
```

Expected: Assertion error will show exactly where the bug is

**Step 3: Fix the specific bug found**

(Details depend on what assertion fails)

---

### Task 7: Integration Test with Fixed Pipeline

**Step 1: Run complete pipeline on 3 frames**

Temporarily modify to process 3 frames:

```python
# In align_with_openpose.py main()
for frame_idx in range(3):  # Was range(25)
```

```python
# In extend_armor_feet.py main()
for frame_idx in range(3):  # Was range(25)
```

**Step 2: Run pipeline**

```bash
python process_clothing_spritesheet.py
```

Expected:
- Should complete armor extension without error
- Should produce offsets (zero or non-zero depending on root cause)
- May fail at mask generation (that's OK for this test)

**Step 3: Verify output frames exist**

```bash
ls -lh training_data/frames_aligned_openpose/
ls -lh training_data/frames_complete_openpose/
```

Expected: Should see 3 frames in each directory

**Step 4: Visual spot check**

Create overlay to verify alignment:

```python
from PIL import Image
base = Image.open('training_data/frames/base_frame_00.png').convert('RGBA')
aligned = Image.open('training_data/frames_aligned_openpose/clothed_frame_00.png').convert('RGBA')
overlay = Image.alpha_composite(base, aligned)
overlay.save('/tmp/alignment_check.png')
```

Open `/tmp/alignment_check.png` and verify shoulders/torso align.

---

### Task 8: Restore Full Frame Processing and Final Test

**Step 1: Restore range(25) in all scripts**

```python
# Restore in align_with_openpose.py and extend_armor_feet.py
for frame_idx in range(25):
```

**Step 2: Remove debug logging**

Remove or comment out all `DEBUG:` print statements from calculate_alignment_offset()

**Step 3: Run complete pipeline**

```bash
python process_clothing_spritesheet.py
```

Expected: Should complete Steps 1-2, may fail at Step 3 (that's a separate issue)

**Step 4: Commit all fixes**

```bash
git add align_with_openpose.py extend_armor_feet.py extract_clothing_final.py tests/
git commit -m "fix: investigate and resolve zero offset issue

Root cause: [FILL IN FROM INVESTIGATION]
Solution: [FILL IN BASED ON SCENARIO A/B/C]

Also fixed return codes in all pipeline scripts to prevent
false failures when checking exit status."
```

---

## Verification Checklist

After completing all tasks:

- [ ] Pipeline doesn't crash with "ERROR: Armor extension failed"
- [ ] Alignment offsets are correct (zero if pre-aligned, non-zero if misaligned)
- [ ] Test test_calculate_alignment_offset passes
- [ ] Visual inspection shows proper alignment (or confirms pre-alignment)
- [ ] All main() functions return 0 on success
- [ ] Debug scripts document findings in /tmp/*.txt files
- [ ] Commits explain root cause and solution

---

## Next Steps

After this plan is complete:
1. Address any remaining pipeline issues (mask generation, etc.)
2. Consider optimization: skip OpenPose if frames are pre-aligned
3. Add proper error handling for OpenPose failures
4. Document alignment requirements in README
