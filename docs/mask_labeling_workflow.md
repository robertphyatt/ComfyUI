# Mask Labeling Workflow

Complete workflow for generating training data for the custom segmentation model.

## Prerequisites

- 25 base frames in `training_data/frames/base_frame_XX.png`
- 25 clothed frames in `training_data/frames/clothed_frame_XX.png`

## Step 1: Generate Initial Masks (Automated)

Run automated color-based mask generation:

```bash
cd /Users/roberthyatt/Code/ComfyUI
python generate_initial_masks.py
```

Output: `training_data/masks_initial/mask_XX.png` (25 files)

This uses color thresholding to identify brown armor pixels vs gray head pixels.

## Step 2: Correct Masks (Manual)

Launch the interactive correction tool:

```bash
python mask_correction_tool.py
```

**Controls:**
- **Left Click:** Add clothing pixels (paint red overlay)
- **Right Click:** Remove clothing pixels (erase red)
- **Mouse Scroll:** Adjust brush size (1-50 pixels)
- **'R' Key:** Reset to initial automated mask
- **'C' Key:** Clear all (start from scratch)
- **Save Button:** Save corrected mask and move to next frame
- **Cancel Button:** Discard changes and exit

**Workflow per frame:**
1. Review the red overlay on the clothed frame
2. Use left click to paint areas that should be clothing
3. Use right click to erase areas that shouldn't be clothing
4. Focus on:
   - Gray head pixels (should NOT be red)
   - Brown armor pixels (SHOULD be red)
   - Edge accuracy around armor boundaries
5. Click Save when satisfied
6. Tool automatically loads next frame

Estimated time: 5-10 minutes per frame = 2-3 hours total

## Step 3: Verify Results

Check corrected masks:

```bash
ls -l training_data/masks_corrected/
# Should show 25 files: mask_00.png through mask_24.png
```

Spot check quality:

```bash
# Open a few masks to verify
open training_data/masks_corrected/mask_00.png
open training_data/masks_corrected/mask_12.png
open training_data/masks_corrected/mask_24.png
```

Look for:
- Clean edges around armor
- No red on gray head
- All armor areas covered

## Output

Final training data structure:

```
training_data/
├── frames/
│   ├── base_frame_00.png ... base_frame_24.png
│   └── clothed_frame_00.png ... clothed_frame_24.png
├── masks_initial/          # Automated generation (reference)
│   └── mask_00.png ... mask_24.png
└── masks_corrected/        # Manual corrections (TRAINING DATA)
    └── mask_00.png ... mask_24.png
```

Use `masks_corrected/` for training the U-Net model.
