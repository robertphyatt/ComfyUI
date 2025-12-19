"""Color correction using golden frame reference.

Maps pixel colors from a golden reference frame to all other frames
using body-segment-based keypoint-relative positioning.
"""

import numpy as np
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class BodySegment(IntEnum):
    """Body segment identifiers."""
    HEAD = 0
    TORSO = 1
    LEFT_UPPER_ARM = 2
    LEFT_LOWER_ARM = 3
    RIGHT_UPPER_ARM = 4
    RIGHT_LOWER_ARM = 5
    LEFT_UPPER_LEG = 6
    LEFT_LOWER_LEG = 7
    RIGHT_UPPER_LEG = 8
    RIGHT_LOWER_LEG = 9


# Keypoint indices for each segment (start, end)
# Based on keypoints.py: 0=head, 1=neck, 2=left_shoulder, 3=right_shoulder,
# 4=left_elbow, 5=right_elbow, 6=left_wrist, 7=right_wrist,
# 8=left_fingertip, 9=right_fingertip, 10=left_hip, 11=right_hip,
# 12=left_knee, 13=right_knee, 14=left_ankle, 15=right_ankle,
# 16=left_toe, 17=right_toe
SEGMENT_KEYPOINTS: Dict[BodySegment, Tuple[int, int]] = {
    BodySegment.HEAD: (0, 1),           # head -> neck
    BodySegment.TORSO: (1, 10),         # neck -> left_hip (use as torso anchor)
    BodySegment.LEFT_UPPER_ARM: (2, 4),  # left_shoulder -> left_elbow
    BodySegment.LEFT_LOWER_ARM: (4, 6),  # left_elbow -> left_wrist
    BodySegment.RIGHT_UPPER_ARM: (3, 5), # right_shoulder -> right_elbow
    BodySegment.RIGHT_LOWER_ARM: (5, 7), # right_elbow -> right_wrist
    BodySegment.LEFT_UPPER_LEG: (10, 12), # left_hip -> left_knee
    BodySegment.LEFT_LOWER_LEG: (12, 14), # left_knee -> left_ankle
    BodySegment.RIGHT_UPPER_LEG: (11, 13), # right_hip -> right_knee
    BodySegment.RIGHT_LOWER_LEG: (13, 15), # right_knee -> right_ankle
}


@dataclass
class PixelPosition:
    """Relative position of a pixel within a body segment."""
    segment: BodySegment
    along_bone: float      # 0.0 = at start keypoint, 1.0 = at end keypoint
    perpendicular: float   # signed distance perpendicular to bone (pixels)


def _point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> Tuple[float, float]:
    """Compute distance from point to line segment.

    Returns:
        (along_bone, perpendicular): Position along bone (0-1) and signed perpendicular distance
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        # Degenerate line segment
        return 0.5, np.linalg.norm(point - line_start)

    line_unit = line_vec / line_len
    point_vec = point - line_start

    # Project point onto line
    along = np.dot(point_vec, line_unit) / line_len  # 0-1 range (can be outside)

    # Perpendicular distance (signed: positive = left of bone direction)
    perp_vec = point_vec - (along * line_len) * line_unit
    perp_dist = np.linalg.norm(perp_vec)

    # Sign: use cross product to determine which side
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    if cross < 0:
        perp_dist = -perp_dist

    return along, perp_dist


def assign_pixel_to_segment(
    pixel_y: int,
    pixel_x: int,
    keypoints: np.ndarray
) -> Optional[PixelPosition]:
    """Assign a pixel to its nearest body segment.

    Args:
        pixel_y: Pixel Y coordinate
        pixel_x: Pixel X coordinate
        keypoints: 18x2 array of keypoint coordinates

    Returns:
        PixelPosition with segment and relative position, or None if too far from any segment
    """
    point = np.array([pixel_x, pixel_y], dtype=np.float64)

    best_segment = None
    best_distance = float('inf')
    best_along = 0.0
    best_perp = 0.0

    for segment, (kp_start, kp_end) in SEGMENT_KEYPOINTS.items():
        start = keypoints[kp_start]
        end = keypoints[kp_end]

        along, perp = _point_to_line_distance(point, start, end)

        # Distance to segment (clamped along to 0-1 for distance calc)
        clamped_along = max(0.0, min(1.0, along))
        closest_on_segment = start + clamped_along * (end - start)
        distance = np.linalg.norm(point - closest_on_segment)

        if distance < best_distance:
            best_distance = distance
            best_segment = segment
            best_along = along
            best_perp = perp

    if best_segment is None:
        return None

    return PixelPosition(
        segment=best_segment,
        along_bone=best_along,
        perpendicular=best_perp
    )


@dataclass
class GoldenPixel:
    """A pixel from the golden frame with its position and color."""
    y: int
    x: int
    rgb: np.ndarray  # (3,) uint8
    position: PixelPosition


def build_golden_index(
    golden_frame: np.ndarray,
    golden_keypoints: np.ndarray
) -> Dict[BodySegment, List[GoldenPixel]]:
    """Build lookup structure for golden frame pixels by segment.

    Args:
        golden_frame: RGBA image of golden frame
        golden_keypoints: 18x2 keypoints for golden frame

    Returns:
        Dict mapping each segment to list of its pixels with positions
    """
    h, w = golden_frame.shape[:2]
    alpha = golden_frame[:, :, 3]

    index: Dict[BodySegment, List[GoldenPixel]] = {seg: [] for seg in BodySegment}

    # Find all visible pixels and assign to segments
    visible_ys, visible_xs = np.where(alpha > 128)

    for y, x in zip(visible_ys, visible_xs):
        position = assign_pixel_to_segment(y, x, golden_keypoints)
        if position is not None:
            pixel = GoldenPixel(
                y=y,
                x=x,
                rgb=golden_frame[y, x, :3].copy(),
                position=position
            )
            index[position.segment].append(pixel)

    return index


def find_golden_color(
    position: PixelPosition,
    golden_index: Dict[BodySegment, List[GoldenPixel]]
) -> Optional[np.ndarray]:
    """Find the color from golden frame for a given relative position.

    Args:
        position: Relative position within a segment
        golden_index: Pre-built index of golden frame

    Returns:
        RGB color (3,) uint8, or None if not found
    """
    segment_pixels = golden_index.get(position.segment, [])

    if not segment_pixels:
        # Segment not visible in golden frame - try nearest segment
        return _find_nearest_segment_color(position, golden_index)

    # Find pixel with closest relative position
    best_pixel = None
    best_distance = float('inf')

    for pixel in segment_pixels:
        # Distance in relative position space
        along_diff = position.along_bone - pixel.position.along_bone
        perp_diff = position.perpendicular - pixel.position.perpendicular

        # Weight along_bone more since it's normalized 0-1
        distance = (along_diff * 50) ** 2 + perp_diff ** 2

        if distance < best_distance:
            best_distance = distance
            best_pixel = pixel

    if best_pixel is None:
        return None

    return best_pixel.rgb


def _find_nearest_segment_color(
    position: PixelPosition,
    golden_index: Dict[BodySegment, List[GoldenPixel]]
) -> Optional[np.ndarray]:
    """Fallback: find color from nearest segment that has pixels."""
    # Try segments in order of likely proximity
    segment_order = [
        BodySegment.TORSO,  # Most likely to have pixels
        BodySegment.LEFT_UPPER_ARM, BodySegment.RIGHT_UPPER_ARM,
        BodySegment.LEFT_UPPER_LEG, BodySegment.RIGHT_UPPER_LEG,
        BodySegment.LEFT_LOWER_ARM, BodySegment.RIGHT_LOWER_ARM,
        BodySegment.LEFT_LOWER_LEG, BodySegment.RIGHT_LOWER_LEG,
        BodySegment.HEAD,
    ]

    for seg in segment_order:
        if seg == position.segment:
            continue
        pixels = golden_index.get(seg, [])
        if pixels:
            # Return color from middle of this segment
            mid_idx = len(pixels) // 2
            return pixels[mid_idx].rgb

    return None


def color_correct_frame(
    frame: np.ndarray,
    frame_keypoints: np.ndarray,
    golden_index: Dict[BodySegment, List[GoldenPixel]]
) -> np.ndarray:
    """Color correct a single frame using golden frame colors.

    Args:
        frame: RGBA image to correct
        frame_keypoints: 18x2 keypoints for this frame
        golden_index: Pre-built index of golden frame

    Returns:
        Color-corrected RGBA image
    """
    result = frame.copy()
    alpha = frame[:, :, 3]

    # Process all visible pixels
    visible_ys, visible_xs = np.where(alpha > 128)

    for y, x in zip(visible_ys, visible_xs):
        # Find this pixel's relative position
        position = assign_pixel_to_segment(y, x, frame_keypoints)

        if position is None:
            continue

        # Find matching color from golden frame
        golden_rgb = find_golden_color(position, golden_index)

        if golden_rgb is not None:
            result[y, x, :3] = golden_rgb

    return result


def color_correct_all(
    frames: List[np.ndarray],
    keypoints_list: List[np.ndarray],
    golden_idx: int
) -> List[np.ndarray]:
    """Color correct all frames using bidirectional propagation from golden frame.

    Instead of correcting every frame from the single golden frame (which fails
    when poses differ significantly), this propagates corrections bidirectionally:
    - Forward: golden → golden+1 → golden+2 → ...
    - Backward: golden → golden-1 → golden-2 → ...

    Each frame uses its already-corrected neighbor as reference, ensuring
    adjacent frames (with similar poses) are compared.

    Args:
        frames: List of RGBA images
        keypoints_list: List of 18x2 keypoint arrays (one per frame)
        golden_idx: Index of the golden frame

    Returns:
        List of color-corrected RGBA images
    """
    if not frames:
        return []

    n_frames = len(frames)
    results: List[Optional[np.ndarray]] = [None] * n_frames

    # Golden frame stays unchanged
    results[golden_idx] = frames[golden_idx].copy()
    print(f"  Frame {golden_idx:02d}: golden (unchanged)")

    # Forward propagation: golden → golden+1 → golden+2 → ... → end
    for i in range(golden_idx + 1, n_frames):
        # Use the previous corrected frame as reference
        ref_idx = i - 1
        ref_frame = results[ref_idx]
        ref_keypoints = keypoints_list[ref_idx]

        # Build index from reference frame
        ref_index = build_golden_index(ref_frame, ref_keypoints)

        # Correct current frame
        corrected = color_correct_frame(frames[i], keypoints_list[i], ref_index)
        results[i] = corrected
        print(f"  Frame {i:02d}: corrected from frame {ref_idx:02d} (forward)")

    # Backward propagation: golden → golden-1 → golden-2 → ... → 0
    for i in range(golden_idx - 1, -1, -1):
        # Use the next corrected frame as reference
        ref_idx = i + 1
        ref_frame = results[ref_idx]
        ref_keypoints = keypoints_list[ref_idx]

        # Build index from reference frame
        ref_index = build_golden_index(ref_frame, ref_keypoints)

        # Correct current frame
        corrected = color_correct_frame(frames[i], keypoints_list[i], ref_index)
        results[i] = corrected
        print(f"  Frame {i:02d}: corrected from frame {ref_idx:02d} (backward)")

    return results
