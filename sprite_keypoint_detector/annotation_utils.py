"""Annotation utilities: retrain model, re-annotate, manual edit.

Three utilities:
1. retrain - Train model on manually confirmed annotations
2. reannotate - Run model on all frames, update only auto keypoints
3. edit - Manual annotation editor with ghost predictions
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import json

from .annotations import load_annotations, save_annotations, is_manual
from .annotator import KeypointAnnotator
from .keypoints import KEYPOINT_NAMES
from .validation import validate_all_annotations

# Import with try/except since these modules may not exist yet
try:
    from .inference import predict_keypoints
except ImportError:
    predict_keypoints = None

try:
    from .train import train_model
except ImportError:
    train_model = None


def retrain_on_manual(
    annotations_path: Path,
    frames_dir: Path,
    model_output_path: Path
) -> None:
    """Retrain keypoint model on manually confirmed annotations only.

    Args:
        annotations_path: Path to annotations JSON
        frames_dir: Directory containing frame images
        model_output_path: Path to save trained model
    """
    if train_model is None:
        raise ImportError("train module not available - cannot retrain model")

    annotations = load_annotations(annotations_path)

    # Filter to only manually annotated keypoints
    manual_annotations = {}
    for frame_name, frame_data in annotations.items():
        keypoints = frame_data.get("keypoints", {})
        manual_kpts = {}
        for name, kp in keypoints.items():
            if isinstance(kp, dict) and kp.get("source") == "manual":
                manual_kpts[name] = [kp["x"], kp["y"]]
            elif isinstance(kp, list):
                # Legacy format - skip (unknown source)
                pass

        if manual_kpts:
            manual_annotations[frame_name] = {
                "image": frame_name,
                "keypoints": manual_kpts
            }

    print(f"Found {len(manual_annotations)} frames with manual annotations")

    if len(manual_annotations) < 5:
        print("Warning: Very few manual annotations. Model may not train well.")

    # Save filtered annotations for training
    train_annotations_path = annotations_path.parent / "train_annotations.json"
    with open(train_annotations_path, 'w') as f:
        json.dump(manual_annotations, f, indent=2)

    # Train model
    train_model(
        annotations_path=train_annotations_path,
        frames_dir=frames_dir,
        output_path=model_output_path
    )

    print(f"Model saved to {model_output_path}")


def reannotate_auto(
    annotations_path: Path,
    frames_dir: Path,
    model_path: Path
) -> None:
    """Re-run automatic annotation, preserving manual keypoints.

    Args:
        annotations_path: Path to annotations JSON
        frames_dir: Directory containing frame images
        model_path: Path to trained model
    """
    if predict_keypoints is None:
        raise ImportError("inference module not available - cannot reannotate")

    annotations = load_annotations(annotations_path)

    # Get all frame images
    frame_files = sorted(frames_dir.glob("*.png"))

    updated_count = 0
    for frame_path in frame_files:
        frame_name = frame_path.name

        # Run inference
        predictions = predict_keypoints(frame_path, model_path)

        # Get existing keypoints
        existing = annotations.get(frame_name, {}).get("keypoints", {})

        # Update only non-manual keypoints
        updated_kpts = {}
        for name in KEYPOINT_NAMES:
            if name in existing:
                kp = existing[name]
                if isinstance(kp, dict) and kp.get("source") == "manual":
                    # Keep manual annotation
                    updated_kpts[name] = kp
                    continue

            # Use prediction
            if name in predictions:
                pred = predictions[name]
                updated_kpts[name] = {
                    "x": pred["x"],
                    "y": pred["y"],
                    "source": "auto",
                    "confidence": pred.get("confidence", 0.5)
                }

        if frame_name not in annotations:
            annotations[frame_name] = {"image": frame_name}

        annotations[frame_name]["keypoints"] = updated_kpts
        updated_count += 1

    save_annotations(annotations, annotations_path)
    print(f"Updated {updated_count} frames with auto predictions (manual preserved)")


def edit_frame(
    frame_path: Path,
    annotations_path: Path,
    model_path: Optional[Path] = None
) -> None:
    """Manually edit annotations for a single frame with ghost predictions.

    Args:
        frame_path: Path to frame image
        annotations_path: Path to annotations JSON
        model_path: Optional model path for ghost predictions
    """
    annotations = load_annotations(annotations_path)
    frame_name = frame_path.name

    existing = annotations.get(frame_name, {}).get("keypoints", {})

    # Get auto predictions for ghost overlay
    auto_predictions = None
    if model_path and model_path.exists():
        if predict_keypoints is None:
            print("Warning: inference module not available - no ghost predictions")
        else:
            auto_predictions = predict_keypoints(frame_path, model_path)

    # Run annotator
    annotator = KeypointAnnotator(frame_path, existing, auto_predictions)
    result = annotator.run()

    if result:
        if frame_name not in annotations:
            annotations[frame_name] = {"image": frame_name}
        annotations[frame_name]["keypoints"] = result
        save_annotations(annotations, annotations_path)
        print(f"Saved annotations for {frame_name}")
    else:
        print("Skipped (no changes saved)")


def edit_flagged(
    annotations_path: Path,
    frames_dir: Path,
    model_path: Optional[Path] = None
) -> None:
    """Edit all flagged frames (low confidence or validation issues).

    Args:
        annotations_path: Path to annotations JSON
        frames_dir: Directory containing frame images
        model_path: Optional model path for ghost predictions
    """
    annotations = load_annotations(annotations_path)
    results = validate_all_annotations(annotations)

    flagged = [r for r in results if not r.is_valid]

    if not flagged:
        print("No flagged frames to edit!")
        return

    print(f"Found {len(flagged)} flagged frames to review")

    for i, result in enumerate(flagged):
        frame_name = result.frame_name
        frame_path = frames_dir / frame_name

        if not frame_path.exists():
            print(f"Warning: {frame_path} not found, skipping")
            continue

        print(f"\n[{i+1}/{len(flagged)}] {frame_name}")
        print(f"  Issues: {result.issues}")
        print(f"  Low confidence: {result.low_confidence_keypoints}")

        resp = input("  Edit this frame? (Y/n/q): ").strip().lower()
        if resp == 'q':
            print("Quitting review")
            break
        if resp == 'n':
            continue

        edit_frame(frame_path, annotations_path, model_path)


def main():
    parser = argparse.ArgumentParser(description="Annotation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Retrain command
    retrain_parser = subparsers.add_parser("retrain", help="Retrain model on manual annotations")
    retrain_parser.add_argument("--annotations", type=Path, required=True)
    retrain_parser.add_argument("--frames", type=Path, required=True)
    retrain_parser.add_argument("--output", type=Path, required=True)

    # Reannotate command
    reann_parser = subparsers.add_parser("reannotate", help="Re-run auto annotation")
    reann_parser.add_argument("--annotations", type=Path, required=True)
    reann_parser.add_argument("--frames", type=Path, required=True)
    reann_parser.add_argument("--model", type=Path, required=True)

    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Manual annotation editor")
    edit_parser.add_argument("--frame", type=Path, help="Single frame to edit")
    edit_parser.add_argument("--flagged", action="store_true", help="Edit all flagged frames")
    edit_parser.add_argument("--annotations", type=Path, required=True)
    edit_parser.add_argument("--frames", type=Path, required=True)
    edit_parser.add_argument("--model", type=Path, help="Model for ghost predictions")

    args = parser.parse_args()

    if args.command == "retrain":
        retrain_on_manual(args.annotations, args.frames, args.output)
    elif args.command == "reannotate":
        reannotate_auto(args.annotations, args.frames, args.model)
    elif args.command == "edit":
        if args.flagged:
            edit_flagged(args.annotations, args.frames, args.model)
        elif args.frame:
            edit_frame(args.frame, args.annotations, args.model)
        else:
            print("Specify --frame or --flagged")


if __name__ == "__main__":
    main()
