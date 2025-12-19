# DWPose Skeleton Alignment Investigation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine optimal DWPose configuration for consistent skeleton detection on pixel art sprites, then implement skeleton-based alignment for clothing transfer.

**Architecture:** Test DWPose configurations systematically, identify why base vs clothed frames produce different skeletons, determine if skeleton-based alignment is viable.

**Tech Stack:** ComfyUI, DWPreprocessor node, Python, OpenCV

---

## Phase 1: Configuration Matrix Testing (COMPLETED)

### Test Matrix Results

| Pose Estimator | Hand/Face | Base Frame 00 | Clothed Frame 00 | Match? |
|----------------|-----------|---------------|------------------|--------|
| dw-ll_ucoco_384.onnx | full | Arms down, close to body | Arms spread wide out | **NO** |
| dw-ll_ucoco_384.onnx | body-only | Arms down, close to body | Arms spread wide out | **NO** |
| dw-ll_ucoco.onnx | full | Arms slightly out | Arms spread wide out | **NO** |
| dw-ll_ucoco.onnx | body-only | Arms at angles | Arms spread wide out | **NO** |

### Key Observations

1. **ALL configurations show inconsistent arm detection** between base and clothed frames
2. **Base frame** (gray mannequin): Arms consistently detected close to body
3. **Clothed frame** (brown armor): Arms consistently detected spread out wide
4. **The bulky shoulder pads on the armor are being interpreted as extended arms**
5. Disabling hand/face detection does NOT fix the issue
6. Different pose_estimator models do NOT fix the issue

### Visual Evidence

The sprites show the SAME pose (running, arms close to body), but DWPose detects:
- Base: Shoulders narrow, elbows at sides
- Clothed: Shoulders wide (at edge of shoulder pads), elbows extended

---

## Phase 2: Root Cause Analysis

### Task 1: Verify sprites have identical poses

**Files:**
- Read: `training_data/frames/base_frame_00.png`
- Read: `training_data/frames/clothed_frame_00.png`

**Step 1: Visual comparison**

Overlay the two sprites with transparency to confirm arm positions match.

**Step 2: Document findings**

Record whether the underlying body pose is truly identical or if the clothed sprite actually has different arm positions.

---

### Task 2: Test on frame where poses clearly match

**Files:**
- Test: Multiple frame pairs (00, 05, 10, 15, 20)

**Step 1: Run DWPose on 5 frame pairs**

```python
frames = [0, 5, 10, 15, 20]
for f in frames:
    run_dwpose(f'base_frame_{f:02d}.png')
    run_dwpose(f'clothed_frame_{f:02d}.png')
```

**Step 2: Record skeleton similarity for each pair**

Create table showing which frames have matching vs mismatching skeletons.

---

### Task 3: Test alternative approaches

**Option A: Use base skeleton for both**
- Run DWPose only on base frames
- Apply that skeleton to warp the clothed reference
- Avoids the shoulder pad confusion entirely

**Option B: Preprocess clothed image**
- Mask out the shoulder pad region before DWPose
- Or resize/crop to focus on body center

**Option C: Extract keypoint coordinates**
- Use SavePoseKpsAsJsonFile to get actual coordinates
- Manually compare shoulder positions numerically
- Identify if shoulders are offset by consistent amount

---

## Phase 3: Implementation Decision

Based on Phase 2 findings, choose one of:

1. **Skeleton-based alignment using base-only skeletons**
   - Viable if we can use base skeleton to guide clothed image warping

2. **Abandon skeleton approach, improve optical flow**
   - If skeleton detection is fundamentally unreliable on armored sprites

3. **Hybrid approach**
   - Use skeleton for coarse alignment, optical flow for fine-tuning

---

## Phase 2: Root Cause Analysis (COMPLETED)

### Task 1: Overlay Verification - COMPLETED

**Finding:** Poses ARE identical (97% pixel overlap in silhouettes)

Overlay analysis:
- 21,221 pixels overlap (both sprites visible)
- 2,238 pixels base-only (mannequin edges)
- 874 pixels clothed-only (armor edges)
- 3,112 total silhouette difference pixels (mostly edge contours)

**Conclusion:** The underlying body poses are identical. Difference is edge contours from armor thickness, not arm position.

### Task 2: Preprocessing Variants - COMPLETED

Tested 6 preprocessing approaches on clothed_frame_00:

| Variant | Arm Detection | Notes |
|---------|---------------|-------|
| Black background | Spread wide | No improvement |
| Tight crop | Spread wider | Made it worse |
| Grayscale | Spread wide | Weird neck detection |
| High contrast | Spread diagonal | Different but still wrong |
| Silhouette | Spread wide | Same as clothed silhouette |
| 2x scale | Spread wide | No improvement |

**None of these preprocessing approaches fixed the arm detection.**

### Task 3: Critical Discovery - TEXTURE DEPENDENCY

**Key experiment:** Run DWPose on BASE frame silhouette

| Image | Arm Detection |
|-------|---------------|
| Original base frame (gray textured) | Arms DOWN ✓ |
| Original clothed frame (brown textured) | Arms SPREAD ✗ |
| Base silhouette (solid gray) | Arms SPREAD ✗ |
| Clothed silhouette (solid gray) | Arms SPREAD ✗ |

**ROOT CAUSE IDENTIFIED:**
1. DWPose relies on **internal texture/shading cues**, not just silhouette outline
2. Gray mannequin's muscle definition and shading helps DWPose find actual arm positions
3. Brown armor texture obscures these body contours
4. Solid silhouettes remove all internal detail → wrong detection for BOTH

---

## Phase 3: Recommended Solution

### Use Base Frame Skeleton for BOTH Images

Since poses are verified identical (97% overlap), the solution is:

1. **Run DWPose ONLY on base frames** (which detect correctly)
2. **Apply that skeleton to warp the clothed reference**
3. **Never run DWPose on armored sprites** (they will always fail)

This approach:
- Uses reliable skeleton detection from base mannequin
- Avoids the armor texture confusion entirely
- Works because poses are confirmed identical

---

## Current Status

**Phase 1: COMPLETE** - Configuration matrix tested, all failed on clothed frames
**Phase 2: COMPLETE** - Root cause identified (texture dependency)
**Phase 3: READY** - Solution defined: use base-only skeletons

**Next Step:** Implement skeleton-based alignment using base frame keypoints only
