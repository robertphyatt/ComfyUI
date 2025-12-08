"""Boundary snapping for mask refinement.

Snaps rough AI segmentation mask boundaries to precise detected edges.
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree


def snap_mask_to_edges(mask_256: np.ndarray, edges_512: np.ndarray, search_radius: int = 10) -> np.ndarray:
    """Snap rough mask boundaries to precise detected edges.

    Upscales the rough 256×256 AI mask to 512×512 using nearest-neighbor,
    then refines boundaries by snapping to nearby detected edges.

    Algorithm:
    1. Upscale mask to 512×512 (nearest-neighbor preserves binary values)
    2. Find all edge pixels (255 in edge map)
    3. Build spatial index (KD-tree) for fast nearest-neighbor lookup
    4. Find mask boundary pixels (pixels where mask transitions 0→1 or 1→0)
    5. For each boundary pixel, find nearest edge within search_radius
    6. Snap boundary pixel to that edge location

    Args:
        mask_256: 256×256 binary mask (0=base, 1=clothing) as uint8 numpy array
        edges_512: 512×512 binary edge map (255=edge, 0=no edge) as uint8 numpy array
        search_radius: Maximum distance to search for nearest edge (default: 10 pixels)

    Returns:
        512×512 refined binary mask (0=base, 1=clothing) as uint8 numpy array

    Raises:
        ValueError: If mask_256 is not 256×256 or edges_512 is not 512×512
        ValueError: If search_radius is not positive
    """
    # Validate inputs
    if mask_256.shape != (256, 256):
        raise ValueError(f"mask_256 must be 256x256, got {mask_256.shape}")
    if edges_512.shape != (512, 512):
        raise ValueError(f"edges_512 must be 512x512, got {edges_512.shape}")
    if search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")

    # Step 1: Upscale mask to 512×512 using nearest-neighbor
    mask_512 = cv2.resize(mask_256, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Step 2: Find all edge pixels
    edge_coords = np.argwhere(edges_512 == 255)  # Returns [[y1, x1], [y2, x2], ...]

    # Early return if no edges detected
    if len(edge_coords) == 0:
        return mask_512  # No edges to snap to, return upscaled mask

    # Step 3: Build KD-tree for fast nearest-neighbor lookup
    edge_tree = cKDTree(edge_coords)

    # Step 4: Find mask boundary pixels
    # Boundaries are pixels where mask value differs from at least one neighbor
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_512, kernel, iterations=1)
    eroded = cv2.erode(mask_512, kernel, iterations=1)
    boundaries = (dilated != eroded).astype(np.uint8)  # Pixels at transitions

    boundary_coords = np.argwhere(boundaries == 1)  # [[y1, x1], [y2, x2], ...]

    # Step 5-6: Snap each boundary pixel to nearest edge within search_radius
    for coord in boundary_coords:
        y, x = coord

        # Query KD-tree for nearest edge
        distance, idx = edge_tree.query(coord, k=1)

        # Only snap if edge is within search radius
        if distance <= search_radius:
            edge_y, edge_x = edge_coords[idx]

            # Determine if this boundary pixel should be clothing or base
            # by checking which side of the transition it's on
            current_value = mask_512[y, x]

            # Snap by updating mask at the edge location to match the boundary value
            # This effectively moves the boundary to the edge
            mask_512[edge_y, edge_x] = current_value

    return mask_512
