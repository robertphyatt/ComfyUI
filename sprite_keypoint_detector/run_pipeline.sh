#!/bin/bash
# Safe pipeline runner that enforces per-animation directory structure
# Usage: ./run_pipeline.sh <animation_name> [--palette-from <other_animation>] [--debug]
#
# Example: ./run_pipeline.sh walk_south --palette-from walk_north --debug

set -e

ANIMATION_NAME="$1"
shift || true

if [ -z "$ANIMATION_NAME" ]; then
    echo "Usage: $0 <animation_name> [--palette-from <other_animation>] [--debug]"
    echo ""
    echo "Available animations:"
    ls -1 training_data/animations/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

ANIMATION_DIR="training_data/animations/${ANIMATION_NAME}"

if [ ! -d "$ANIMATION_DIR" ]; then
    echo "ERROR: Animation directory not found: $ANIMATION_DIR"
    echo ""
    echo "Available animations:"
    ls -1 training_data/animations/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Verify required files exist
if [ ! -f "$ANIMATION_DIR/annotations.json" ]; then
    echo "ERROR: annotations.json not found in $ANIMATION_DIR"
    exit 1
fi

FRAME_COUNT=$(ls "$ANIMATION_DIR/frames/"*.png 2>/dev/null | wc -l | tr -d ' ')
MASK_COUNT=$(ls "$ANIMATION_DIR/masks/"*.png 2>/dev/null | wc -l | tr -d ' ')

if [ "$FRAME_COUNT" -eq 0 ]; then
    echo "ERROR: No frames found in $ANIMATION_DIR/frames/"
    exit 1
fi

if [ "$MASK_COUNT" -eq 0 ]; then
    echo "ERROR: No masks found in $ANIMATION_DIR/masks/"
    exit 1
fi

echo "=== Running pipeline for: $ANIMATION_NAME ==="
echo "Frames: $FRAME_COUNT"
echo "Masks: $MASK_COUNT"
echo ""

# Build command
CMD="python -m sprite_keypoint_detector.pipeline"
CMD="$CMD --frames-dir $ANIMATION_DIR/frames"
CMD="$CMD --annotations $ANIMATION_DIR/annotations.json"
CMD="$CMD --masks $ANIMATION_DIR/masks"
CMD="$CMD --output $ANIMATION_DIR/output"
CMD="$CMD --skip-validation"

# Process additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --palette-from)
            OTHER_ANIMATION="$2"
            PALETTE_DIR="training_data/animations/${OTHER_ANIMATION}/output"
            if [ ! -d "$PALETTE_DIR" ]; then
                echo "ERROR: Palette source not found: $PALETTE_DIR"
                exit 1
            fi
            CMD="$CMD --palette-from $PALETTE_DIR"
            shift 2
            ;;
        --debug)
            CMD="$CMD --debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running: $CMD"
echo ""
$CMD
