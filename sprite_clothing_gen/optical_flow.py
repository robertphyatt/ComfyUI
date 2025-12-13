"""Optical flow warping for sprite clothing transfer."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def load_image_bgr(path: Path) -> np.ndarray:
    """Load image as BGR numpy array for OpenCV processing."""
    img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def save_image_bgr(arr: np.ndarray, path: Path) -> None:
    """Save BGR numpy array as image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    img.save(path)
