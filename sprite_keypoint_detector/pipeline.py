"""Main clothing spritesheet pipeline.

Usage with spritesheets:
    python -m sprite_keypoint_detector.pipeline \
        --base base_spritesheet.png \
        --reference clothed_spritesheet.png \
        --annotations annotations.json \
        --masks masks/ \
        --output output_dir/

Usage with frame directories:
    python -m sprite_keypoint_detector.pipeline \
        --frames-dir frames/ \
        --annotations annotations.json \
        --masks masks/ \
        --output output_dir/

Frame directory structure:
    frames/
        base_frame_00.png
        base_frame_01.png
        ...
        clothed_frame_00.png
        clothed_frame_01.png
        ...
"""

import argparse
import cv2
import numpy as np
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

from .annotations import load_annotations, save_annotations, get_coords_array
from .validation import validate_all_annotations, ValidationResult
from .matching import find_top_candidates, score_candidate_after_transform, select_best_match, MatchCandidate, FrameMatch, compute_joint_distance
from .transform import transform_frame, transform_frame_debug, get_keypoints_array, TransformConfig, TransformDebugOutput, apply_pixelize
from .spritesheet import (
    detect_layout, split_spritesheet, assemble_spritesheet,
    save_frames, composite_overlay, SpritesheetLayout
)
from .keypoints import KEYPOINT_NAMES
from .golden_selection import select_golden_frame
from .color_correction import color_correct_all


def create_debug_comparison(debug_dir: Path, frame_idx: int) -> np.ndarray:
    """Create a side-by-side comparison of all debug steps for a frame.

    Args:
        debug_dir: Debug output directory
        frame_idx: Frame index

    Returns:
        Composite image with all steps side by side, labeled
    """
    steps = [
        ("1_aligned", "Aligned"),
        ("2_masked", "Masked"),
        ("3_rotated", "Rotated"),
        ("4_inpainted", "Inpainted"),
        ("5_color_corrected", "Color Corrected"),
        ("6_final", "Final"),
        ("overlap", "Overlap"),
        ("skeleton", "Skeleton"),
    ]

    images = []
    for folder, label in steps:
        path = debug_dir / folder / f"frame_{frame_idx:02d}.png"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Failed to load debug image {path}")
                continue
            # Convert to BGRA if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # Add label at top
            label_height = 30
            labeled = np.zeros((img.shape[0] + label_height, img.shape[1], 4), dtype=np.uint8)
            labeled[:, :] = [40, 40, 40, 255]  # Dark gray background
            labeled[label_height:, :] = img

            # Draw label text
            cv2.putText(labeled, label, (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 1)

            images.append(labeled)

    if not images:
        return None

    # Stack horizontally
    return np.hstack(images)


def load_frames_from_directory(frames_dir: Path, prefix: str) -> List[np.ndarray]:
    """Load frames from directory matching pattern prefix_XX.png.

    Args:
        frames_dir: Directory containing frame images
        prefix: Prefix to match (e.g., 'base_frame' or 'clothed_frame')

    Returns:
        List of frames sorted by index
    """
    pattern = re.compile(rf"{prefix}_(\d+)\.png$")
    frame_files = []

    for f in frames_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            idx = int(match.group(1))
            frame_files.append((idx, f))

    # Sort by index
    frame_files.sort(key=lambda x: x[0])

    frames = []
    for idx, path in frame_files:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError(f"Failed to load frame: {path}")
        frames.append(frame)

    return frames


class ClothingPipeline:
    """Main pipeline for generating clothing spritesheets."""

    def __init__(
        self,
        annotations_path: Path,
        masks_dir: Path,
        output_dir: Path,
        config: Optional[TransformConfig] = None,
        base_spritesheet_path: Optional[Path] = None,
        reference_spritesheet_path: Optional[Path] = None,
        frames_dir: Optional[Path] = None
    ):
        """Initialize pipeline from either spritesheets or frame directory.

        Args:
            annotations_path: Path to annotations JSON
            masks_dir: Directory containing armor masks
            output_dir: Output directory
            config: Transform configuration
            base_spritesheet_path: Path to base spritesheet (spritesheet mode)
            reference_spritesheet_path: Path to clothed spritesheet (spritesheet mode)
            frames_dir: Directory containing individual frames (directory mode)
        """
        self.annotations_path = Path(annotations_path)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.config = config or TransformConfig()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)

        # Load annotations
        self.annotations = load_annotations(self.annotations_path)

        # Load frames from either spritesheets or directory
        if frames_dir is not None:
            # Directory mode: load individual frames
            frames_dir = Path(frames_dir)
            print(f"Loading frames from directory: {frames_dir}")

            self.base_frames = load_frames_from_directory(frames_dir, "base_frame")
            self.reference_frames = load_frames_from_directory(frames_dir, "clothed_frame")

            if len(self.base_frames) != len(self.reference_frames):
                raise ValueError(f"Frame count mismatch: {len(self.base_frames)} base frames vs {len(self.reference_frames)} clothed frames")

            print(f"Loaded {len(self.base_frames)} base frames, {len(self.reference_frames)} reference frames")

            # Create layout from frame dimensions (assume 5x5 grid for 25 frames)
            if self.base_frames:
                h, w = self.base_frames[0].shape[:2]
                n_frames = len(self.base_frames)
                # Calculate grid: prefer square-ish layout
                cols = int(np.ceil(np.sqrt(n_frames)))
                rows = int(np.ceil(n_frames / cols))
                self.layout = SpritesheetLayout(
                    frame_width=w,
                    frame_height=h,
                    columns=cols,
                    rows=rows,
                    total_frames=n_frames
                )
                print(f"Layout for assembly: {cols}x{rows} grid, {w}x{h}px frames")
        else:
            # Spritesheet mode: split spritesheets
            if base_spritesheet_path is None or reference_spritesheet_path is None:
                raise ValueError("Must provide either frames_dir or both base_spritesheet_path and reference_spritesheet_path")

            self.base_path = Path(base_spritesheet_path)
            self.reference_path = Path(reference_spritesheet_path)

            self.base_spritesheet = cv2.imread(str(self.base_path), cv2.IMREAD_UNCHANGED)
            self.reference_spritesheet = cv2.imread(str(self.reference_path), cv2.IMREAD_UNCHANGED)

            # Detect layout from base
            self.layout = detect_layout(self.base_spritesheet)
            print(f"Detected layout: {self.layout.columns}x{self.layout.rows} frames, "
                  f"{self.layout.frame_width}x{self.layout.frame_height}px each")

            # Split spritesheets
            self.base_frames = split_spritesheet(self.base_spritesheet, self.layout)
            self.reference_frames = split_spritesheet(self.reference_spritesheet, self.layout)

            print(f"Split {len(self.base_frames)} base frames, {len(self.reference_frames)} reference frames")

    def validate_annotations(self) -> List[ValidationResult]:
        """Validate all annotations, return flagged frames."""
        print("\n=== Validating Annotations ===")
        results = validate_all_annotations(self.annotations)

        flagged = [r for r in results if not r.is_valid]
        print(f"Total frames: {len(results)}")
        print(f"Flagged for review: {len(flagged)}")

        for r in flagged:
            print(f"  {r.frame_name}:")
            for issue in r.issues:
                print(f"    - {issue}")
            for lc in r.low_confidence_keypoints:
                print(f"    - Low confidence: {lc}")

        return flagged

    def match_frames(self, red_threshold: int = 2000) -> List[FrameMatch]:
        """Match each base frame to best clothed frame."""
        print("\n=== Matching Frames ===")

        # Separate base and clothed annotations
        base_annotations = {k: v for k, v in self.annotations.items() if k.startswith("base_")}
        clothed_annotations = {k: v for k, v in self.annotations.items() if k.startswith("clothed_")}

        matches = []

        for base_idx, (base_name, base_data) in enumerate(sorted(base_annotations.items())):
            print(f"\nMatching {base_name} ({base_idx + 1}/{len(base_annotations)})")

            base_kpts = base_data.get("keypoints", {})
            base_frame = self.base_frames[base_idx]

            # Find top 3 candidates by joint distance
            candidates = find_top_candidates(base_name, base_kpts, clothed_annotations)
            print(f"  Top 3 by joint distance: {[c[0] for c in candidates]}")

            # Score each candidate after transform
            scored_candidates = []
            for clothed_name, joint_dist in candidates:
                clothed_idx = int(clothed_name.split("_")[-1].replace(".png", ""))
                clothed_frame = self.reference_frames[clothed_idx]
                clothed_kpts = clothed_annotations[clothed_name].get("keypoints", {})

                # Load mask
                mask_path = self.masks_dir / f"mask_{clothed_idx:02d}.png"
                if not mask_path.exists():
                    print(f"    Warning: mask not found for {clothed_name}")
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                # Transform and score
                clothed_kpts_array = get_keypoints_array(clothed_kpts)
                base_kpts_array = get_keypoints_array(base_kpts)

                transformed = transform_frame(
                    clothed_frame, clothed_kpts_array,
                    base_frame, base_kpts_array,
                    mask, self.config
                )

                neck_y = int(base_kpts_array[1, 1])
                blue, red = score_candidate_after_transform(base_frame, transformed, neck_y)

                scored_candidates.append(MatchCandidate(
                    clothed_frame=clothed_name,
                    joint_distance=joint_dist,
                    blue_pixels=blue,
                    red_pixels=red,
                    score_rank=0
                ))
                print(f"    {clothed_name}: blue={blue}, red={red}")

            # Select best
            if scored_candidates:
                best, needs_review = select_best_match(scored_candidates, red_threshold)

                # Update ranks (red first, then blue - matches selection logic)
                sorted_by_score = sorted(scored_candidates, key=lambda c: (c.red_pixels, c.blue_pixels))
                for rank, c in enumerate(sorted_by_score):
                    c.score_rank = rank + 1

                match = FrameMatch(
                    base_frame=base_name,
                    matched_clothed_frame=best.clothed_frame,
                    candidates=scored_candidates,
                    needs_review=needs_review
                )
                matches.append(match)

                status = "NEEDS REVIEW" if needs_review else "OK"
                print(f"  -> Best match: {best.clothed_frame} (red={best.red_pixels}, blue={best.blue_pixels}) [{status}]")

        return matches

    def generate_outputs(self, matches: List[FrameMatch], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate final clothing and debug overlay spritesheets.

        Args:
            matches: List of frame matches
            debug: If True, save intermediate step outputs for each frame
        """
        print("\n=== Generating Outputs ===")

        clothed_annotations = {k: v for k, v in self.annotations.items() if k.startswith("clothed_")}
        base_annotations = {k: v for k, v in self.annotations.items() if k.startswith("base_")}

        # Collect inpainted frames and keypoints for color correction
        inpainted_frames = []
        frame_keypoints = []
        frame_indices = []  # Track base_idx for each frame for debug output

        # Create debug directories if needed
        debug_dir = None
        if debug:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            (debug_dir / "1_aligned").mkdir(exist_ok=True)
            (debug_dir / "2_masked").mkdir(exist_ok=True)
            (debug_dir / "3_rotated").mkdir(exist_ok=True)
            (debug_dir / "4_inpainted").mkdir(exist_ok=True)
            (debug_dir / "5_color_corrected").mkdir(exist_ok=True)
            (debug_dir / "6_final").mkdir(exist_ok=True)
            (debug_dir / "overlap").mkdir(exist_ok=True)
            (debug_dir / "skeleton").mkdir(exist_ok=True)

        for match in matches:
            base_idx = int(match.base_frame.split("_")[-1].replace(".png", ""))
            clothed_idx = int(match.matched_clothed_frame.split("_")[-1].replace(".png", ""))

            base_frame = self.base_frames[base_idx]
            clothed_frame = self.reference_frames[clothed_idx]

            base_kpts = get_keypoints_array(base_annotations[match.base_frame].get("keypoints", {}))
            clothed_kpts = get_keypoints_array(clothed_annotations[match.matched_clothed_frame].get("keypoints", {}))

            # Load mask
            mask_path = self.masks_dir / f"mask_{clothed_idx:02d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            # Compute joint distance to determine if poses are similar enough to skip rotation
            # Low joint distance means poses nearly match - rotation would only add noise artifacts
            joint_dist = compute_joint_distance(
                base_annotations[match.base_frame].get("keypoints", {}),
                clothed_annotations[match.matched_clothed_frame].get("keypoints", {}),
                KEYPOINT_NAMES
            )

            # Skip rotation if poses are nearly identical (joint distance < threshold)
            # Note: 18 keypoints, so ~8px average error per keypoint = ~150 total
            ROTATION_SKIP_THRESHOLD = 150  # Total joint distance in pixels across all keypoints
            skip_rotation = joint_dist < ROTATION_SKIP_THRESHOLD

            # Create per-frame config using dataclasses.replace()
            frame_config = replace(self.config, skip_rotation=skip_rotation)

            if skip_rotation:
                print(f"    (skipping rotation - joint_dist={joint_dist:.1f} < {ROTATION_SKIP_THRESHOLD})")

            if debug:
                # Use debug transform to get all intermediate steps
                debug_output = transform_frame_debug(
                    clothed_frame, clothed_kpts,
                    base_frame, base_kpts,
                    mask, frame_config
                )

                # Save intermediate outputs (up to inpainted - color correction and final come later)
                cv2.imwrite(str(debug_dir / "1_aligned" / f"frame_{base_idx:02d}.png"), debug_output.aligned_clothed)
                cv2.imwrite(str(debug_dir / "2_masked" / f"frame_{base_idx:02d}.png"), debug_output.armor_masked)
                cv2.imwrite(str(debug_dir / "3_rotated" / f"frame_{base_idx:02d}.png"), debug_output.rotated_armor)
                cv2.imwrite(str(debug_dir / "4_inpainted" / f"frame_{base_idx:02d}.png"), debug_output.inpainted_armor)
                cv2.imwrite(str(debug_dir / "overlap" / f"frame_{base_idx:02d}.png"), debug_output.overlap_viz)
                cv2.imwrite(str(debug_dir / "skeleton" / f"frame_{base_idx:02d}.png"), debug_output.skeleton_viz)

                inpainted_frames.append(debug_output.final_armor)
            else:
                # Normal transform
                transformed = transform_frame(
                    clothed_frame, clothed_kpts,
                    base_frame, base_kpts,
                    mask, frame_config
                )
                inpainted_frames.append(transformed)

            # Store keypoints for color correction (use base keypoints since armor is aligned to base)
            frame_keypoints.append(base_kpts)
            frame_indices.append(base_idx)

            print(f"  Generated frame {base_idx:02d} from {match.matched_clothed_frame}")

        # === Golden frame selection and color correction ===
        print("\n=== Golden Frame Selection ===")
        print("Select the best frame as golden reference...")

        # Convert BGR to RGB for display in matplotlib GUI
        display_frames = []
        for frame in inpainted_frames:
            if frame.shape[2] == 4:
                # BGRA -> RGBA
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            else:
                # BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frames.append(rgb)

        # Show GUI for golden selection
        golden_idx = select_golden_frame(display_frames)
        print(f"Selected frame {frame_indices[golden_idx]:02d} as golden reference")

        # Color correct all frames
        print("\n=== Applying Color Correction ===")
        corrected_frames = color_correct_all(inpainted_frames, frame_keypoints, golden_idx)

        # Save color-corrected frames to debug if enabled
        if debug:
            for i, (corrected, base_idx) in enumerate(zip(corrected_frames, frame_indices)):
                cv2.imwrite(str(debug_dir / "5_color_corrected" / f"frame_{base_idx:02d}.png"), corrected)
            print(f"  Saved color-corrected frames to debug/5_color_corrected/")

        # Apply pixelization as final step
        print("\n=== Applying Pixelization ===")
        final_frames = [apply_pixelize(f, self.config.pixelize_factor) for f in corrected_frames]

        # Save final frames to debug if enabled
        if debug:
            for i, (final, base_idx) in enumerate(zip(final_frames, frame_indices)):
                cv2.imwrite(str(debug_dir / "6_final" / f"frame_{base_idx:02d}.png"), final)
            print(f"  Saved final frames to debug/6_final/")

        # Save individual frames
        save_frames(final_frames, self.output_dir / "frames", prefix="clothing")

        # Assemble clothing spritesheet
        clothing_sheet = assemble_spritesheet(final_frames, self.layout)
        cv2.imwrite(str(self.output_dir / "clothing.png"), clothing_sheet)
        print(f"Saved: {self.output_dir / 'clothing.png'}")

        # Create debug overlay
        overlay_frames = composite_overlay(self.base_frames, final_frames)
        overlay_sheet = assemble_spritesheet(overlay_frames, self.layout)
        cv2.imwrite(str(self.output_dir / "debug_overlay.png"), overlay_sheet)
        print(f"Saved: {self.output_dir / 'debug_overlay.png'}")

        if debug:
            # Create side-by-side comparison images
            comparison_dir = debug_dir / "comparison"
            comparison_dir.mkdir(exist_ok=True)

            for match in matches:
                base_idx = int(match.base_frame.split("_")[-1].replace(".png", ""))
                comparison = create_debug_comparison(debug_dir, base_idx)
                if comparison is not None:
                    cv2.imwrite(str(comparison_dir / f"frame_{base_idx:02d}.png"), comparison)

            print(f"Saved debug outputs to: {debug_dir}")
            print(f"View comparisons: open {debug_dir / 'comparison'}")

        return clothing_sheet, overlay_sheet

    def run(self, skip_validation: bool = False, debug: bool = False) -> bool:
        """Run the full pipeline.

        Args:
            skip_validation: Skip annotation validation step
            debug: Save intermediate step outputs for each frame

        Returns:
            True if successful, False if manual intervention needed
        """
        # Step 1: Validate annotations
        if not skip_validation:
            flagged = self.validate_annotations()
            if flagged:
                print(f"\n{len(flagged)} frames need manual review before proceeding.")
                print("Run with --skip-validation to proceed anyway, or fix annotations first.")
                return False

        # Step 2: Match frames
        matches = self.match_frames()

        # Check for frames needing review
        needs_review = [m for m in matches if m.needs_review]
        if needs_review:
            print(f"\n{len(needs_review)} matches need manual review:")
            for m in needs_review:
                print(f"  {m.base_frame} -> {m.matched_clothed_frame}")
            print("Proceeding with best available matches...")

        # Step 3: Generate outputs
        self.generate_outputs(matches, debug=debug)

        print("\n=== Pipeline Complete ===")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate clothing spritesheet from base and reference"
    )

    # Input mode: either spritesheets or frame directory
    parser.add_argument("--frames-dir", type=Path,
                       help="Directory containing base_frame_XX.png and clothed_frame_XX.png")
    parser.add_argument("--base", type=Path,
                       help="Base mannequin spritesheet (requires --reference)")
    parser.add_argument("--reference", type=Path,
                       help="Clothed reference spritesheet (required with --base)")
    parser.add_argument("--annotations", type=Path, required=True,
                       help="Annotations JSON file")
    parser.add_argument("--masks", type=Path, required=True,
                       help="Directory containing armor masks")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip annotation validation")
    parser.add_argument("--scale", type=float, default=1.057,
                       help="Scale factor for clothed frames")
    parser.add_argument("--pixelize", type=int, default=3,
                       help="Pixelization factor (1=none)")
    parser.add_argument("--debug", action="store_true",
                       help="Save intermediate step outputs for each frame")

    args = parser.parse_args()

    # Validate input mode: either frames-dir OR both base+reference
    if args.frames_dir is None and (args.base is None or args.reference is None):
        parser.error("Must provide either --frames-dir OR both --base and --reference")
    if args.frames_dir is not None and (args.base is not None or args.reference is not None):
        parser.error("Cannot use --frames-dir with --base/--reference")

    config = TransformConfig(
        scale_factor=args.scale,
        pixelize_factor=args.pixelize
    )

    pipeline = ClothingPipeline(
        annotations_path=args.annotations,
        masks_dir=args.masks,
        output_dir=args.output,
        config=config,
        base_spritesheet_path=args.base,
        reference_spritesheet_path=args.reference,
        frames_dir=args.frames_dir
    )

    success = pipeline.run(skip_validation=args.skip_validation, debug=args.debug)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
