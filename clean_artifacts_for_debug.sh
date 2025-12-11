#!/bin/bash
# Clean all pipeline artifacts except source files

echo "Cleaning pipeline artifacts..."

# Remove generated frames
rm -rf training_data/frames_ipadapter_generated/
rm -rf training_data/frames_complete_ipadapter/

# Remove old debug outputs
rm -rf output/debug/
rm -f output/ipadapter_generated_*.png
rm -f output/pose_*.png

# Remove validation data
rm -rf training_data_validation/

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "✓ Artifacts cleaned"
echo "Preserved: examples/input/, training_data/frames/, training_data/masks_inpainting/"
