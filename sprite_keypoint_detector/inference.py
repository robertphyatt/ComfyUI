"""Inference utilities for sprite keypoint detector."""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2

from .model import SpriteKeypointDetector, VIEW_ANGLES
from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS, SKELETON_CONNECTIONS, SKELETON_COLORS


def get_view_angle_index(view_angle: str) -> int:
    """Convert view angle string to index."""
    view_angle = view_angle.lower()
    if view_angle in VIEW_ANGLES:
        return VIEW_ANGLES.index(view_angle)
    # Try to extract from filename
    for idx, view in enumerate(VIEW_ANGLES):
        if view in view_angle:
            return idx
    return 0  # Default to north


class SpriteKeypointPredictor:
    """Wrapper for easy inference with trained model."""

    def __init__(self, model_path: Path, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        self.model = SpriteKeypointDetector(num_keypoints=NUM_KEYPOINTS, use_view_conditioning=True)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {model_path}")
        if 'val_error_px' in checkpoint:
            print(f"  Validation error: {checkpoint['val_error_px']:.1f}px")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: Union[Path, Image.Image, np.ndarray],
                view_angle: str = "north") -> Dict[str, Tuple[int, int]]:
        """Predict keypoints for an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            view_angle: View angle string ('north', 'south', 'east', 'west', etc.)
                       or filename containing view angle (e.g., 'walk_south_frame01.png')
        """
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        # Get view angle index
        view_str = image_path if image_path else view_angle
        view_idx = get_view_angle_index(view_str)
        view_tensor = torch.tensor([view_idx], dtype=torch.long).to(self.device)

        orig_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            keypoints_normalized = self.model(image_tensor, view_tensor)[0]

        keypoints_px = keypoints_normalized.cpu().numpy()
        keypoints_px[:, 0] *= orig_size[0]
        keypoints_px[:, 1] *= orig_size[1]

        result = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            result[name] = (int(round(keypoints_px[i, 0])), int(round(keypoints_px[i, 1])))

        return result


def draw_skeleton(
    image: Union[Image.Image, np.ndarray],
    keypoints: Dict[str, Tuple[int, int]],
    point_radius: int = 5,
    line_thickness: int = 2
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    result = image.copy()
    kp_list = [keypoints.get(name) for name in KEYPOINT_NAMES]

    for (i, j), color in zip(SKELETON_CONNECTIONS, SKELETON_COLORS):
        if kp_list[i] is not None and kp_list[j] is not None:
            cv2.line(result, kp_list[i], kp_list[j], color, line_thickness)

    for i, kp in enumerate(kp_list):
        if kp is not None:
            color = SKELETON_COLORS[min(i, len(SKELETON_COLORS)-1)]
            cv2.circle(result, kp, point_radius, color, -1)
            cv2.circle(result, kp, point_radius, (255, 255, 255), 1)

    return result


def render_skeleton_only(
    keypoints: Dict[str, Tuple[int, int]],
    image_size: Tuple[int, int] = (512, 512),
    point_radius: int = 5,
    line_thickness: int = 2,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    result = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
    return draw_skeleton(result, keypoints, point_radius, line_thickness)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m sprite_keypoint_detector.inference <model.pth> <image.png>")
        sys.exit(1)

    predictor = SpriteKeypointPredictor(Path(sys.argv[1]))
    keypoints = predictor.predict(Path(sys.argv[2]))

    print("Detected keypoints:")
    for name, (x, y) in keypoints.items():
        print(f"  {name}: ({x}, {y})")

    image = Image.open(sys.argv[2]).convert('RGB')
    result = draw_skeleton(image, keypoints)
    output_path = Path(sys.argv[2]).parent / f"{Path(sys.argv[2]).stem}_skeleton.png"
    Image.fromarray(result).save(output_path)
    print(f"\nSaved: {output_path}")
