# Palette-Based Color Synchronization Design

**Date:** 2025-12-19
**Status:** Ready for implementation

## Problem

Body-segment keypoint-relative color correction fails when poses differ between frames. Even with bidirectional propagation, mapping pixels by relative position on skeleton bones causes:
- Black legs (failed lookups when pose differs)
- Wrong colors (mapping to shadow/edge pixels)
- Color drift across the animation

## Solution

SNES-style 16-color palette extraction and remapping. Instead of position-aware color mapping, extract an optimal palette from all frames and quantize everything to use only those colors.

## Flow

1. Run existing pipeline (align → mask → rotate → inpaint)
2. Extract optimal 16-color palette from ALL inpainted frames using k-means
3. Remap every pixel in every frame to nearest palette color
4. Apply pixelization
5. Output final spritesheet

## Implementation

### Palette Extraction

```python
def extract_palette(frames: List[np.ndarray], n_colors: int = 16) -> np.ndarray:
    """Extract optimal n-color palette from all frames using k-means."""
    # Collect all visible pixels
    all_pixels = []
    for frame in frames:
        mask = frame[:, :, 3] > 128
        rgb = frame[:, :, :3][mask]
        all_pixels.append(rgb)

    all_pixels = np.vstack(all_pixels)

    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(all_pixels)

    palette = kmeans.cluster_centers_.astype(np.uint8)
    return palette
```

### Frame Remapping

```python
def remap_frame_to_palette(frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Remap all visible pixels in frame to nearest palette color."""
    result = frame.copy()
    mask = frame[:, :, 3] > 128
    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        pixel_rgb = frame[y, x, :3]
        distances = np.sqrt(np.sum((palette - pixel_rgb) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        result[y, x, :3] = palette[nearest_idx]

    return result


def remap_all_frames(frames: List[np.ndarray], palette: np.ndarray) -> List[np.ndarray]:
    """Remap all frames to use the shared palette."""
    return [remap_frame_to_palette(f, palette) for f in frames]
```

### Pipeline Integration

```python
# After all frames are inpainted, before pixelization:

print("\n=== Extracting Color Palette ===")
palette = extract_palette(inpainted_frames, n_colors=16)
print(f"  Extracted {len(palette)}-color palette")

print("\n=== Remapping Frames to Palette ===")
corrected_frames = remap_all_frames(inpainted_frames, palette)
print(f"  Remapped {len(corrected_frames)} frames")

# Apply pixelization as final step
final_frames = [apply_pixelize(f, self.config.pixelize_factor) for f in corrected_frames]
```

### Debug Output

- `5_palette_remapped/` - frames after palette remapping
- `palette.png` - visual 4x4 grid of the 16-color palette

## Files to Modify

- `color_correction.py` - Replace body-segment code with palette functions
- `pipeline.py` - Remove golden selection, use palette functions
- `golden_selection.py` - Delete (no longer needed)

## Dependencies

- `scikit-learn` for k-means clustering

## Benefits

- No golden frame selection needed (simpler UX)
- All frames guaranteed to use identical colors
- SNES-authentic aesthetic
- Much simpler code than body-segment approach
- No pose-dependency issues
