"""Spritesheet utilities: split into frames, assemble from frames."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpritesheetLayout:
    """Layout information for a spritesheet."""
    frame_width: int
    frame_height: int
    columns: int
    rows: int
    total_frames: int


def detect_layout(spritesheet: np.ndarray) -> SpritesheetLayout:
    """Detect spritesheet layout by analyzing frame boundaries.

    Assumes:
    - All frames are same size
    - Frames are arranged in a grid
    - Transparent gaps between frames (or frame edges are detectable)

    Args:
        spritesheet: RGBA spritesheet image

    Returns:
        SpritesheetLayout with detected dimensions
    """
    h, w = spritesheet.shape[:2]
    alpha = spritesheet[:, :, 3]

    # Find vertical gaps (columns with mostly transparent pixels)
    col_alpha = np.mean(alpha, axis=0)

    # Find horizontal gaps (rows with mostly transparent pixels)
    row_alpha = np.mean(alpha, axis=1)

    # Detect frame boundaries by finding transitions
    # A frame boundary is where alpha goes from low to high or high to low
    threshold = 10  # Alpha threshold for "transparent"

    # Find frame width by detecting vertical boundaries
    in_frame = col_alpha > threshold
    transitions = np.diff(in_frame.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        # Fallback: assume single row, detect by aspect ratio
        # Common sprite sizes: 64x64, 128x128, 256x256
        for size in [64, 128, 256, 512]:
            if w % size == 0 and h % size == 0:
                cols = w // size
                rows = h // size
                return SpritesheetLayout(
                    frame_width=size,
                    frame_height=size,
                    columns=cols,
                    rows=rows,
                    total_frames=cols * rows
                )
        # Last resort: assume square frames based on height
        return SpritesheetLayout(
            frame_width=h,
            frame_height=h,
            columns=w // h,
            rows=1,
            total_frames=w // h
        )

    # Estimate frame width from detected boundaries
    frame_widths = ends - starts + 1
    if len(frame_widths) > 0:
        frame_width = int(np.median(frame_widths))
    else:
        frame_width = w

    # Similarly for height
    in_frame_row = row_alpha > threshold
    transitions_row = np.diff(in_frame_row.astype(int))
    starts_row = np.where(transitions_row == 1)[0] + 1
    ends_row = np.where(transitions_row == -1)[0]

    if len(ends_row) > 0 and len(starts_row) > 0:
        frame_heights = ends_row - starts_row + 1
        frame_height = int(np.median(frame_heights))
    else:
        frame_height = h

    columns = max(1, w // frame_width)
    rows = max(1, h // frame_height)

    return SpritesheetLayout(
        frame_width=frame_width,
        frame_height=frame_height,
        columns=columns,
        rows=rows,
        total_frames=columns * rows
    )


def split_spritesheet(
    spritesheet: np.ndarray,
    layout: Optional[SpritesheetLayout] = None
) -> List[np.ndarray]:
    """Split spritesheet into individual frames.

    Args:
        spritesheet: RGBA spritesheet image
        layout: Optional layout (auto-detected if None)

    Returns:
        List of frame images in row-major order
    """
    if layout is None:
        layout = detect_layout(spritesheet)

    frames = []
    for row in range(layout.rows):
        for col in range(layout.columns):
            x = col * layout.frame_width
            y = row * layout.frame_height
            frame = spritesheet[y:y+layout.frame_height, x:x+layout.frame_width].copy()
            frames.append(frame)

    return frames


def assemble_spritesheet(
    frames: List[np.ndarray],
    layout: SpritesheetLayout
) -> np.ndarray:
    """Assemble frames into a spritesheet.

    Args:
        frames: List of frame images
        layout: Layout specifying grid arrangement

    Returns:
        Assembled spritesheet image
    """
    sheet_h = layout.rows * layout.frame_height
    sheet_w = layout.columns * layout.frame_width

    # Determine channels from first frame
    if len(frames[0].shape) == 3:
        channels = frames[0].shape[2]
        spritesheet = np.zeros((sheet_h, sheet_w, channels), dtype=np.uint8)
    else:
        spritesheet = np.zeros((sheet_h, sheet_w), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        if idx >= layout.total_frames:
            break
        row = idx // layout.columns
        col = idx % layout.columns
        x = col * layout.frame_width
        y = row * layout.frame_height
        spritesheet[y:y+layout.frame_height, x:x+layout.frame_width] = frame

    return spritesheet


def save_frames(
    frames: List[np.ndarray],
    output_dir: Path,
    prefix: str = "frame"
) -> List[Path]:
    """Save individual frames to directory.

    Args:
        frames: List of frame images
        output_dir: Directory to save frames
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for idx, frame in enumerate(frames):
        path = output_dir / f"{prefix}_{idx:02d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)

    return paths


def load_frames(
    frame_dir: Path,
    pattern: str = "frame_*.png"
) -> List[np.ndarray]:
    """Load frames from directory.

    Args:
        frame_dir: Directory containing frames
        pattern: Glob pattern for frame files

    Returns:
        List of frame images sorted by filename
    """
    frame_dir = Path(frame_dir)
    paths = sorted(frame_dir.glob(pattern))

    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        frames.append(frame)

    return frames


def composite_overlay(
    base_frames: List[np.ndarray],
    overlay_frames: List[np.ndarray]
) -> List[np.ndarray]:
    """Composite overlay frames on top of base frames.

    Args:
        base_frames: List of base frame images
        overlay_frames: List of overlay frame images (same length as base)

    Returns:
        List of composited frames
    """
    composites = []

    for base, overlay in zip(base_frames, overlay_frames):
        result = base.copy()

        # Alpha composite
        overlay_alpha = overlay[:, :, 3:4] / 255.0
        result[:, :, :3] = (
            result[:, :, :3] * (1 - overlay_alpha) +
            overlay[:, :, :3] * overlay_alpha
        ).astype(np.uint8)
        result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])

        composites.append(result)

    return composites
