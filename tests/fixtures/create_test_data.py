"""Create test fixtures for integration testing."""

from pathlib import Path
from PIL import Image, ImageDraw


def create_test_spritesheet(output_path: Path, grid_size=(5, 5), frame_size=64):
    """Create a test 5x5 spritesheet with distinct frames.

    Each frame has a unique color and number overlay for identification.
    """
    cols, rows = grid_size
    sheet_width = frame_size * cols
    sheet_height = frame_size * rows

    # Create base image
    img = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    frame_index = 0
    for row in range(rows):
        for col in range(cols):
            # Calculate position
            x = col * frame_size
            y = row * frame_size

            # Create unique color for this frame
            r = (frame_index * 10) % 256
            g = (frame_index * 15) % 256
            b = (frame_index * 20) % 256

            # Draw frame background
            draw.rectangle(
                [x, y, x + frame_size, y + frame_size],
                fill=(r, g, b, 255)
            )

            # Draw frame number
            text = str(frame_index)
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x + (frame_size - text_width) // 2
            text_y = y + (frame_size - text_height) // 2
            draw.text((text_x, text_y), text, fill=(255, 255, 255, 255))

            frame_index += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def create_test_reference_frame(output_path: Path, frame_size=64):
    """Create a test reference frame with 'clothing'.

    Simple frame with colored body and clothing overlay.
    """
    img = Image.new('RGBA', (frame_size, frame_size), (100, 100, 200, 255))
    draw = ImageDraw.Draw(img)

    # Draw "clothing" as red rectangle in center
    padding = frame_size // 4
    draw.rectangle(
        [padding, padding, frame_size - padding, frame_size - padding],
        fill=(200, 50, 50, 255)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


if __name__ == '__main__':
    # Generate test fixtures
    fixtures_dir = Path(__file__).parent
    create_test_spritesheet(fixtures_dir / 'test_spritesheet.png')
    create_test_reference_frame(fixtures_dir / 'test_reference.png')
    print("✅ Test fixtures created")
