"""Semantic segmentation-based clothing extraction.

Extracts clothing from sprite frames using AI semantic segmentation
combined with edge detection for precise boundary refinement.
"""

from PIL import Image
import numpy as np

from ai_segmentation import call_ollama_segmentation
from edge_detection import detect_clothing_edges
from boundary_snapping import snap_mask_to_edges
from morphological_cleanup import cleanup_mask


def extract_clothing_semantic(clothed_frame: Image.Image, base_frame: Image.Image) -> Image.Image:
    """Extract clothing using semantic segmentation with edge refinement.

    Pipeline:
    1. Downscale clothed frame to 256×256 for AI processing
    2. AI semantic segmentation: classify each pixel as clothing vs base
    3. Edge detection: find precise boundaries in frame difference
    4. Boundary snapping: refine AI mask to detected edges
    5. Morphological cleanup: fill holes, remove islands
    6. Apply mask: create transparent image with only clothing visible

    Args:
        clothed_frame: PIL Image (512×512) of character wearing clothing
        base_frame: PIL Image (512×512) of base character without clothing

    Returns:
        PIL Image (512×512 RGBA) with clothing on transparent background

    Raises:
        ValueError: If images are not 512×512 or don't match dimensions
    """
    # Validate inputs
    if clothed_frame.size != (512, 512):
        raise ValueError(f"clothed_frame must be 512x512, got {clothed_frame.size}")
    if base_frame.size != (512, 512):
        raise ValueError(f"base_frame must be 512x512, got {base_frame.size}")
    if clothed_frame.size != base_frame.size:
        raise ValueError(f"Frame dimensions must match: {clothed_frame.size} vs {base_frame.size}")

    print("Step 1/6: Downscaling for AI processing...")
    # Step 1: Downscale clothed frame to 256×256
    clothed_256 = clothed_frame.resize((256, 256), Image.LANCZOS)

    print("Step 2/6: AI semantic segmentation...")
    # Step 2: AI segmentation (returns 256×256 binary mask)
    mask_256 = call_ollama_segmentation(clothed_256)

    print("Step 3/6: Edge detection...")
    # Step 3: Edge detection on frame difference
    edges_512 = detect_clothing_edges(clothed_frame, base_frame)

    print("Step 4/6: Boundary snapping...")
    # Step 4: Snap mask boundaries to detected edges
    mask_512_snapped = snap_mask_to_edges(mask_256, edges_512, search_radius=10)

    print("Step 5/6: Morphological cleanup...")
    # Step 5: Morphological cleanup
    mask_512_cleaned = cleanup_mask(mask_512_snapped, close_iterations=2, open_iterations=1)

    print("Step 6/6: Applying final mask...")
    # Step 6: Apply mask to clothed frame
    clothed_arr = np.array(clothed_frame.convert('RGBA'))

    # Create output with transparency
    clothing_arr = clothed_arr.copy()
    clothing_arr[:, :, 3] = np.where(
        mask_512_cleaned == 1,
        clothed_arr[:, :, 3],  # Keep clothing (preserve original alpha)
        0                       # Remove base character (transparent)
    )

    return Image.fromarray(clothing_arr, 'RGBA')
