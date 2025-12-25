# Clothing Mask Prediction Model

## Overview

This directory contains a trained U-Net model for predicting clothing masks from sprite animation frames. The model takes a (base_body, clothed_body) image pair and outputs a binary mask showing which pixels belong to the clothing.

## Why This Exists

When transforming clothing from one character pose to another, we need accurate masks to isolate just the clothing pixels. Simple image differencing (thresholding `|clothed - base|`) only achieves ~81% F1 score because:
- Subtle color differences between skin and clothing get missed
- Shadows and lighting variations cause false positives/negatives
- Edge cases where clothing color matches skin tones

The U-Net model achieves **96.6% F1 score** by learning these patterns from manually corrected training data.

## Model Architecture

- **Type:** Small U-Net (encoder-decoder with skip connections)
- **Input:** 8 channels (base RGBA + clothed RGBA), resized to 256x256
- **Output:** 1 channel binary mask, resized back to original dimensions
- **Loss:** Combined Dice + BCE loss (weighted 70% Dice, 30% BCE) to handle class imbalance
- **Training:** 200 epochs on MPS/CUDA/CPU

## Training Data

50 manually corrected mask samples from two animations:
- `frames_backup_walk_north/` + `masks_initial_backup_walk_north/` (25 samples)
- `frames_backup_walk_south/` + `masks_backup_walk_south/` (25 samples)

Each sample consists of:
- `base_frame_XX.png` - The nude/base body sprite (512x512 RGBA)
- `clothed_frame_XX.png` - The same pose with clothing (512x512 RGBA)
- `mask_XX.png` - Binary mask of clothing pixels (512x512 grayscale)

## Files

- `mask_model.py` - Model definition, training, and inference code
- `mask_model.pth` - Trained model weights
- `mask_correction_tool.py` - Interactive tool for manually correcting masks

## Usage

### Predicting masks for new frames:

```python
from mask_model import load_model, predict_mask

model, device = load_model('mask_model.pth')
mask = predict_mask(model, 'path/to/base.png', 'path/to/clothed.png', device)
# mask is a numpy array (H, W) with values 0 or 255
```

### Training on new data:

1. Split your spritesheet into individual frames (base + clothed pairs)
2. Generate initial masks using the model or simple thresholding
3. Correct masks manually using `mask_correction_tool.py`
4. Add frames to a backup directory following the naming convention
5. Update `gather_training_data()` in `mask_model.py` to include new data
6. Retrain: `python mask_model.py training_data`

### Manual mask correction workflow:

```bash
# Set up directories
cp your_frames/*.png training_data/frames/
# Generate initial masks (or copy from model predictions)
cp initial_masks/*.png training_data/masks_initial/

# Run correction tool
python mask_correction_tool.py training_data

# Tool outputs corrected masks to training_data/masks_corrected/
```

## Key Learnings

1. **Dice loss is essential** - BCE alone causes massive underprediction due to class imbalance (most pixels are background). Dice loss directly optimizes for overlap.

2. **Simple thresholding caps at ~81% F1** - No matter how you tune the threshold, you can't capture the nuances that a learned model can.

3. **Iterative correction works** - Start with model predictions, manually correct problem frames, retrain, repeat until convergence.

4. **Backup your training data** - Keep separate backup directories for each animation (e.g., `frames_backup_walk_north/`) so you don't accidentally overwrite when switching between datasets.

## Performance

| Dataset | Avg F1 | Min | Max |
|---------|--------|-----|-----|
| Walk South (25 frames) | 0.960 | 0.944 | 0.970 |
| Walk North (25 frames) | 0.972 | 0.951 | 0.979 |
| **Overall** | **0.966** | | |

## Dependencies

- PyTorch (with MPS/CUDA support recommended)
- OpenCV (cv2)
- NumPy
- matplotlib (for mask_correction_tool.py)
