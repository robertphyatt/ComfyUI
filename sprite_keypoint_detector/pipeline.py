"""Main clothing spritesheet pipeline.

Usage:
    python -m sprite_keypoint_detector.pipeline \
        --base base_spritesheet.png \
        --reference clothed_spritesheet.png \
        --output output_dir/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from .annotations import load_annotations, save_annotations, get_coords_array
from .validation import validate_all_annotations, ValidationResult
from .matching import find_top_candidates, score_candidate_after_transform, select_best_match, MatchCandidate, FrameMatch
from .transform import transform_frame, get_keypoints_array, TransformConfig
from .spritesheet import (
    detect_layout, split_spritesheet, assemble_spritesheet,
    save_frames, composite_overlay, SpritesheetLayout
)
from .keypoints import KEYPOINT_NAMES


class ClothingPipeline:
    """Main pipeline for generating clothing spritesheets."""

    def __init__(
        self,
        base_spritesheet_path: Path,
        reference_spritesheet_path: Path,
        annotations_path: Path,
        masks_dir: Path,
        output_dir: Path,
        config: Optional[TransformConfig] = None
    ):
        self.base_path = Path(base_spritesheet_path)
        self.reference_path = Path(reference_spritesheet_path)
        self.annotations_path = Path(annotations_path)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.config = config or TransformConfig()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)

        # Load data
        self.annotations = load_annotations(self.annotations_path)
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

    def match_frames(self, blue_threshold: int = 2000) -> List[FrameMatch]:
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

            # Find top 5 candidates by joint distance
            candidates = find_top_candidates(base_name, base_kpts, clothed_annotations, top_n=5)
            print(f"  Top 5 by joint distance: {[c[0] for c in candidates]}")

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
                best, needs_review = select_best_match(scored_candidates, blue_threshold)

                # Update ranks
                sorted_by_score = sorted(scored_candidates, key=lambda c: (c.blue_pixels, c.red_pixels))
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
                print(f"  -> Best match: {best.clothed_frame} (blue={best.blue_pixels}) [{status}]")

        return matches

    def generate_outputs(self, matches: List[FrameMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate final clothing and debug overlay spritesheets."""
        print("\n=== Generating Outputs ===")

        clothed_annotations = {k: v for k, v in self.annotations.items() if k.startswith("clothed_")}
        base_annotations = {k: v for k, v in self.annotations.items() if k.startswith("base_")}

        clothing_frames = []

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

            # Transform
            transformed = transform_frame(
                clothed_frame, clothed_kpts,
                base_frame, base_kpts,
                mask, self.config
            )

            clothing_frames.append(transformed)
            print(f"  Generated frame {base_idx:02d} from {match.matched_clothed_frame}")

        # Save individual frames
        save_frames(clothing_frames, self.output_dir / "frames", prefix="clothing")

        # Assemble clothing spritesheet
        clothing_sheet = assemble_spritesheet(clothing_frames, self.layout)
        cv2.imwrite(str(self.output_dir / "clothing.png"), clothing_sheet)
        print(f"Saved: {self.output_dir / 'clothing.png'}")

        # Create debug overlay
        overlay_frames = composite_overlay(self.base_frames, clothing_frames)
        overlay_sheet = assemble_spritesheet(overlay_frames, self.layout)
        cv2.imwrite(str(self.output_dir / "debug_overlay.png"), overlay_sheet)
        print(f"Saved: {self.output_dir / 'debug_overlay.png'}")

        return clothing_sheet, overlay_sheet

    def run(self, skip_validation: bool = False) -> bool:
        """Run the full pipeline.

        Args:
            skip_validation: Skip annotation validation step

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
        self.generate_outputs(matches)

        print("\n=== Pipeline Complete ===")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate clothing spritesheet from base and reference"
    )
    parser.add_argument("--base", type=Path, required=True,
                       help="Base mannequin spritesheet")
    parser.add_argument("--reference", type=Path, required=True,
                       help="Clothed reference spritesheet")
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

    args = parser.parse_args()

    config = TransformConfig(
        scale_factor=args.scale,
        pixelize_factor=args.pixelize
    )

    pipeline = ClothingPipeline(
        base_spritesheet_path=args.base,
        reference_spritesheet_path=args.reference,
        annotations_path=args.annotations,
        masks_dir=args.masks,
        output_dir=args.output,
        config=config
    )

    success = pipeline.run(skip_validation=args.skip_validation)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
