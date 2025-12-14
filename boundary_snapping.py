"""Boundary snapping for mask refinement.

Snaps rough AI segmentation mask boundaries to precise detected edges.
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree


def snap_mask_to_edges(mask_128: np.ndarray, edges_512: np.ndarray, search_radius: int = 10) -> np.ndarray:
    """Snap rough mask boundaries to precise detected edges.

    Upscales the rough 128×128 AI mask to 512×512 using nearest-neighbor,
    then refines boundaries by snapping to nearby detected edges.

    Algorithm:
    1. Upscale mask to 512×512 (nearest-neighbor preserves binary values)
    2. Find all edge pixels (255 in edge map)
    3. Build spatial index (KD-tree) for fast nearest-neighbor lookup
    4. Find mask boundary pixels (pixels where mask transitions 0→1 or 1→0)
    5. For each boundary pixel, find nearest edge within search_radius
    6. Snap boundary pixel to that edge location

    Args:
        mask_128: 128×128 binary mask (0=base, 1=clothing) as uint8 numpy array
        edges_512: 512×512 binary edge map (255=edge, 0=no edge) as uint8 numpy array
        search_radius: Maximum distance to search for nearest edge (default: 10 pixels)

    Returns:
        512×512 refined binary mask (0=base, 1=clothing) as uint8 numpy array

    Raises:
        ValueError: If mask_128 is not 128×128 or edges_512 is not 512×512
        ValueError: If search_radius is not positive
    """
    # Validate inputs
    if mask_128.shape != (128, 128):
        raise ValueError(f"mask_128 must be 128x128, got {mask_128.shape}")
    if edges_512.shape != (512, 512):
        raise ValueError(f"edges_512 must be 512x512, got {edges_512.shape}")
    if search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")

    # Step 1: Upscale mask to 512×512 using nearest-neighbor
    mask_512 = cv2.resize(mask_128, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Step 2: Find all edge pixels
    edge_coords = np.argwhere(edges_512 == 255)

    # Early return if no edges detected
    if len(edge_coords) == 0:
        return mask_512

    # Step 3: Build KD-tree for fast nearest-neighbor lookup
    edge_tree = cKDTree(edge_coords)

    # Create working copy for modifications
    result_mask = mask_512.copy()

    # Step 4a: Find inner boundaries on original mask and expand them
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_512, kernel, iterations=1)
    eroded = cv2.erode(mask_512, kernel, iterations=1)

    # Inner boundary: clothing pixels (1) at the edge (has neighbors that are 0)
    inner_boundary = ((mask_512 == 1) & (dilated != eroded)).astype(np.uint8)
    inner_coords = np.argwhere(inner_boundary == 1)

    # Step 5a: Snap inner boundaries (expand mask outward to edges)
    if len(inner_coords) > 0:
        # Vectorized nearest-neighbor query
        distances, indices = edge_tree.query(inner_coords, k=1)
        valid_snaps = distances <= search_radius

        # For each valid snap, draw line from boundary to edge
        for i in np.where(valid_snaps)[0]:
            boundary_y, boundary_x = inner_coords[i]
            edge_y, edge_x = edge_coords[indices[i]]

            # Draw line from boundary to edge, filling with 1 (clothing)
            cv2.line(result_mask, (boundary_x, boundary_y), (edge_x, edge_y),
                     color=1, thickness=1)

    # Step 4b: Find outer boundaries on the EXPANDED mask (after inner snapping)
    # This ensures we don't erase pixels we just added
    dilated = cv2.dilate(result_mask, kernel, iterations=1)
    eroded = cv2.erode(result_mask, kernel, iterations=1)

    # Outer boundary: base pixels (0) at the edge (has neighbors that are 1)
    outer_boundary = ((result_mask == 0) & (dilated != eroded)).astype(np.uint8)
    outer_coords = np.argwhere(outer_boundary == 1)

    # Step 5b: Snap outer boundaries (contract mask inward to edges)
    if len(outer_coords) > 0:
        # Vectorized nearest-neighbor query
        distances, indices = edge_tree.query(outer_coords, k=1)
        valid_snaps = distances <= search_radius

        # For each valid snap, draw line from edge to boundary
        for i in np.where(valid_snaps)[0]:
            boundary_y, boundary_x = outer_coords[i]
            edge_y, edge_x = edge_coords[indices[i]]

            # Draw line from edge to boundary, filling with 0 (base)
            cv2.line(result_mask, (edge_x, edge_y), (boundary_x, boundary_y),
                     color=0, thickness=1)

    return result_mask
