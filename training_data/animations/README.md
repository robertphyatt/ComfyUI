# Animation Training Data

Each animation direction has its own self-contained directory to prevent data corruption from overwrites.

## Directory Structure

```
animations/
├── walk_north/
│   ├── frames/           # base_frame_XX.png + clothed_frame_XX.png
│   ├── masks/            # mask_XX.png (manually corrected)
│   ├── annotations.json  # Skeleton keypoints for all frames
│   └── output/           # Pipeline output (generated)
├── walk_south/
├── walk_east/
└── walk_west/
```

## Workflow

### 1. Extract frames from spritesheet
```bash
python -m sprite_keypoint_detector.extract_frames \
    --spritesheet training_data/spritesheets/base.png \
    --output training_data/animations/walk_south/frames/ \
    --prefix base_frame
```

### 2. Annotate skeletons
```bash
python -m sprite_keypoint_detector.annotator \
    training_data/animations/walk_south/frames/ \
    training_data/animations/walk_south/annotations.json
```

### 3. Create/correct masks
```bash
python -m sprite_keypoint_detector.mask_correction_tool \
    training_data/animations/walk_south/
```

### 4. COMMIT IMMEDIATELY after manual work
```bash
git add training_data/animations/walk_south/
git commit -m "feat: add walk_south frames, masks, and annotations"
```

### 5. Run pipeline (using safe wrapper)
```bash
./sprite_keypoint_detector/run_pipeline.sh walk_south --palette-from walk_north --debug
```

## Rules

1. **NEVER** use `training_data/frames/` - it's deprecated and gitignored
2. **ALWAYS** commit frames, masks, and annotations immediately after creation
3. **NEVER** overwrite another animation's directory
4. Each animation is self-contained and isolated
5. Use the wrapper script `run_pipeline.sh` instead of calling pipeline.py directly

## Why This Structure?

On 2025-12-25, we lost hours of manual mask work because:
- The shared `training_data/frames/` directory was overwritten by a different operation
- Masks drawn for north-facing frames were used with south-facing frames
- The mismatch caused the pipeline to produce garbage output

This per-animation structure prevents that by isolating each animation's data.

## Troubleshooting

**"No frames found" error:**
- Check that frames are in `training_data/animations/<name>/frames/`
- Not in the old `training_data/frames/` location

**"Palette source not found" error:**
- The animation you're copying palette from must have been run first
- Check `training_data/animations/<name>/output/` exists

**Masks don't match frames:**
- Verify you're using frames and masks for the same animation direction
- Each mask should be drawn on the corresponding clothed frame
