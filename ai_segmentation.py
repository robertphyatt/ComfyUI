# ai_segmentation.py
"""AI-powered semantic segmentation using Ollama vision model."""
import json
import base64
import io
import numpy as np
import requests
from PIL import Image
from typing import Dict, Any
from rle_utils import decode_rle

def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL image as base64 string.

    Args:
        image: PIL Image (must be 256x256)

    Raises:
        ValueError: If image is not 256x256
    """
    if image.size != (256, 256):
        raise ValueError(f"Image must be 256x256, got {image.size}")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_ollama_segmentation(clothed_frame_256: Image.Image) -> np.ndarray:
    """Call Ollama to generate semantic segmentation mask.

    Args:
        clothed_frame_256: 256x256 clothed character frame

    Returns:
        256x256 binary mask (0=base character, 1=clothing)
    """
    # Encode image
    image_b64 = encode_image_base64(clothed_frame_256)

    # Build prompt
    prompt = """Analyze this 256×256 pixel sprite showing a character wearing clothing/armor.

Classify each pixel as either:
- CLOTHING (1): New armor/clothing pixels (brown leather armor, equipment, etc.)
- BASE (0): Original gray character pixels showing through (gray skin/head visible through helmet)
- TRANSPARENT pixels (alpha < 50): Classify as BASE (0)

Focus on RGB color values, not alpha channel. Transparent regions are background, not clothing.

Output as run-length encoding to compress the 65,536 pixel mask:
{
  "mask": [
    {"value": 0, "count": 1234},
    {"value": 1, "count": 567},
    ...
  ]
}

Rules:
- Start at top-left pixel, proceed row-by-row
- Group consecutive pixels with same value
- All runs must sum to exactly 65,536 pixels

Output ONLY valid JSON, no other text."""

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "ministral-3:8b",
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }

    try:
        print("   Calling Ollama for semantic segmentation...")
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to call Ollama API: {e}\nURL: {url}") from e

    result = response.json()
    response_text = result.get("response", "")

    # Parse RLE
    try:
        rle_data = json.loads(response_text)
        rle_mask = rle_data["mask"]
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to parse Ollama response: {e}\nResponse: {response_text}")

    # Decode RLE to flat array
    flat_mask = decode_rle(rle_mask, length=256*256)

    # Reshape to 2D
    mask_2d = flat_mask.reshape((256, 256))

    print(f"   ✓ Segmentation complete: {np.sum(mask_2d == 1)} clothing pixels, {np.sum(mask_2d == 0)} base pixels")

    return mask_2d
