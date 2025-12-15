"""Annotation schema and utilities for keypoint metadata tracking."""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

# New schema: each keypoint has x, y, source, confidence
# {
#   "frame.png": {
#     "image": "frame.png",
#     "keypoints": {
#       "head": {"x": 249, "y": 172, "source": "manual", "confidence": 1.0},
#       "neck": {"x": 252, "y": 217, "source": "auto", "confidence": 0.92}
#     }
#   }
# }


def migrate_legacy_annotation(keypoints: Dict) -> Dict:
    """Convert legacy [x, y] format to new {x, y, source, confidence} format.

    Legacy format: {"head": [249, 172], "neck": [252, 217]}
    New format: {"head": {"x": 249, "y": 172, "source": "auto", "confidence": 0.0}}
    """
    migrated = {}
    for name, value in keypoints.items():
        if isinstance(value, list):
            # Legacy format - assume auto with unknown confidence
            migrated[name] = {
                "x": value[0],
                "y": value[1],
                "source": "auto",
                "confidence": 0.0  # Unknown confidence from legacy
            }
        elif isinstance(value, dict) and "x" in value:
            # Already new format
            migrated[name] = value
        else:
            raise ValueError(f"Unknown keypoint format for {name}: {value}")
    return migrated


def get_keypoint_coords(keypoint: Dict) -> Tuple[int, int]:
    """Extract (x, y) coordinates from keypoint dict."""
    return (keypoint["x"], keypoint["y"])


def is_manual(keypoint: Dict) -> bool:
    """Check if keypoint was manually annotated."""
    return keypoint.get("source") == "manual"


def create_keypoint(x: int, y: int, source: str = "manual", confidence: float = 1.0) -> Dict:
    """Create a keypoint entry with full metadata."""
    return {"x": x, "y": y, "source": source, "confidence": confidence}


def load_annotations(path: Path) -> Dict[str, Dict]:
    """Load annotations, migrating legacy format if needed."""
    with open(path) as f:
        data = json.load(f)

    # Migrate each frame's keypoints if needed
    for frame_name, frame_data in data.items():
        if "keypoints" in frame_data:
            # Check if legacy format (first keypoint is a list)
            sample = next(iter(frame_data["keypoints"].values()), None)
            if isinstance(sample, list):
                frame_data["keypoints"] = migrate_legacy_annotation(frame_data["keypoints"])

    return data


def save_annotations(data: Dict[str, Dict], path: Path) -> None:
    """Save annotations to JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_coords_array(keypoints: Dict, keypoint_names: List[str]) -> List[Tuple[int, int]]:
    """Extract coordinate list from keypoints dict in order of keypoint_names."""
    coords = []
    for name in keypoint_names:
        if name in keypoints:
            kp = keypoints[name]
            if isinstance(kp, list):
                coords.append((kp[0], kp[1]))
            else:
                coords.append((kp["x"], kp["y"]))
        else:
            coords.append((0, 0))  # Missing keypoint
    return coords
