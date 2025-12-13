"""Optical flow warping for sprite clothing transfer."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def load_image_bgr(path: Path) -> np.ndarray:
    """Load image as BGR numpy array for OpenCV processing."""
    img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
