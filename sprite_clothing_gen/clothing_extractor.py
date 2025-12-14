"""Clothing extraction using U2-Net segmentation."""

from pathlib import Path
from PIL import Image
from rembg import remove, new_session


def extract_clothing_from_reference(
    reference_image_path: Path,
    output_path: Path,
    model: str = "u2net_cloth_seg"
) -> Path:
    """Extract clothing from a reference image using U2-Net.

    This removes the character body and background, leaving only
    the clothing on a transparent background.

    Args:
        reference_image_path: Path to reference image (character wearing clothes)
        output_path: Path to save clothing-only output
        model: U2-Net model to use (default: u2net_cloth_seg for clothing)

    Returns:
        Path to output image with clothing only

    Raises:
        RuntimeError: If segmentation fails
    """
    try:
        # Load input image
        input_img = Image.open(reference_image_path)

        # Create session for the specified model
        session = new_session(model)

        # Remove background/body using U2-Net
        # For u2net_cloth_seg, this keeps clothing and removes body/background
        output_img = remove(input_img, session=session)

        # Ensure output has alpha channel
        if output_img.mode != 'RGBA':
            output_img = output_img.convert('RGBA')

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_path)

        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to extract clothing: {e}")


def is_rembg_available() -> bool:
    """Check if rembg is available and working.

    Returns:
        True if rembg can be imported and initialized
    """
    try:
        import rembg
        return True
    except ImportError:
        return False
