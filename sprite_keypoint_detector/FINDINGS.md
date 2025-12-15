# Sprite Skeleton Warping: Findings and Methodology

## Overview

This document describes a pipeline for overlaying armor/clothing from reference sprites onto base mannequin sprites using skeleton-based Thin-Plate-Spline (TPS) warping with greedy optimization.

## Problem Statement

We want to create a "paper doll" style character system where armor/clothing from one spritesheet can be overlaid onto a base mannequin spritesheet. The challenge is that the armor sprite and base sprite have different poses, requiring intelligent warping to make the armor conform to the base skeleton.

## Pipeline Steps

### 1. Skeleton Keypoint Detection

We use an 18-keypoint skeleton model trained on annotated sprite frames:

```
Keypoints (18 total):
- 0: head
- 1: neck
- 2-3: left_shoulder, right_shoulder
- 4-5: left_elbow, right_elbow
- 6-7: left_wrist, right_wrist
- 8-9: left_fingertip, right_fingertip
- 10-11: left_hip, right_hip
- 12-13: left_knee, right_knee
- 14-15: left_ankle, right_ankle
- 16-17: left_toe, right_toe
```

The model uses a ResNet18 backbone (frozen) with a trainable head, trained on 27 annotated frames.

### 2. Scale and Alignment

Before warping, we:
1. Calculate a **scale factor** by comparing average body heights (head-to-ankle) between base and clothed sprites
2. **Scale** the clothed sprite to match the base sprite size
3. **Align** by neck position so both skeletons share the same neck location

Current scale factor: **0.9504** (clothed sprites are ~5% larger and need to scale down)

### 3. Armor Extraction

We use pre-generated masks to isolate just the armor from the clothed sprite, removing the underlying body/head.

### 4. TPS Warp

Thin-Plate-Spline warping uses the skeleton keypoints as control points to deform the armor image so it conforms to the base skeleton pose.

### 5. Greedy Optimization

The key innovation is a greedy hill-climbing optimizer that adjusts the clothed skeleton keypoints to minimize "gray pixel leakage" (uncovered base pixels showing through the armor).

**Algorithm:**
1. Start with detected skeleton keypoints
2. For each optimizable keypoint, try moving it 1px in each direction (up/down/left/right)
3. After each move, run the full TPS warp and count uncovered gray pixels
4. Keep any move that reduces uncovered pixels
5. Repeat until no improvement is possible

**Constraints:**
- Head and neck are never moved (indices 0, 1)
- Fingertips and toes are never moved independently (indices 8, 9, 16, 17)
- Gray pixels **above the neck Y position** don't count (prevents optimizer from warping armor up to cover the head)

## Critical Discovery: Extremity Coupling

### The Problem

When fingertips and toes were optimized independently from their parent joints (wrists and ankles), the TPS warp created severe distortion artifacts:
- Arms would bend at unnatural angles (wrist position conflicting with fingertip position)
- Feet would stretch and distort (ankle moving opposite to toe)

### The Solution

**Extremity coupling**: When a parent joint moves, its child extremity moves with it by the same amount.

```python
EXTREMITY_PAIRS = {
    6: 8,   # left_wrist -> left_fingertip
    7: 9,   # right_wrist -> right_fingertip
    14: 16, # left_ankle -> left_toe
    15: 17, # right_ankle -> right_toe
}
```

This maintains the natural limb shape while still allowing the optimizer to adjust positions.

## Results

With the 18-keypoint model and extremity coupling:

| Frame | Before | After | Reduction |
|-------|--------|-------|-----------|
| clothed_frame_00 | 1828 px | 429 px | 77% |
| clothed_frame_01 | 1773 px | 249 px | 86% |

## Key Files

- `keypoints.py` - Keypoint definitions (18 points), skeleton connections, colors
- `optimizer.py` - TPS warp, scale/align, greedy optimization algorithm
- `train.py` - Model training script
- `model.py` - ResNet18-based keypoint detection model
- `dataset.py` - Data loading and augmentation

## Lessons Learned

1. **Extremities need control points** - Without fingertip/toe keypoints, the TPS warp has no anchor at limb ends, causing unpredictable stretching.

2. **Extremities must follow parents** - Independent optimization of extremities creates conflicting control points that distort the warp.

3. **Neck constraint prevents cheating** - Without ignoring pixels above the neck, the optimizer would warp armor upward to cover the head area.

4. **Mask alignment matters** - The armor extraction mask must match the target base frame. Using masks designed for a different frame causes complete failure.

## Future Work

- Automate mask generation for new clothed sprites
- Extend to all 25 animation frames
- Add pixel-art post-processing to clean up warp artifacts
- Support multiple armor sets per character
