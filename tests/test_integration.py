"""Integration tests for sprite clothing generation."""

import pytest
from pathlib import Path
from sprite_clothing_gen.orchestrator import SpriteClothingGenerator
from sprite_clothing_gen.comfy_client import ComfyUIClient
from sprite_clothing_gen.config import COMFYUI_URL


@pytest.fixture
def test_spritesheet():
    """Path to test spritesheet fixture."""
    return Path(__file__).parent / "fixtures" / "test_spritesheet.png"


@pytest.fixture
def test_reference():
    """Path to test reference frame fixture."""
    return Path(__file__).parent / "fixtures" / "test_reference.png"


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    return tmp_path


@pytest.mark.integration
@pytest.mark.skipif(
    not ComfyUIClient(COMFYUI_URL).health_check(),
    reason="ComfyUI server not running"
)
def test_full_pipeline(test_spritesheet, test_reference, output_dir):
    """Test complete pipeline from spritesheet to clothing output.

    This test requires ComfyUI server to be running.
    Run with: pytest -m integration
    """

    generator = SpriteClothingGenerator()

    output_path = output_dir / "output_clothing.png"

    result = generator.generate(
        base_spritesheet=test_spritesheet,
        reference_frame=test_reference,
        reference_frame_index=12,  # Middle frame
        output_path=output_path,
        seed=42,
        keep_temp=True  # Keep for inspection
    )

    # Verify output exists
    assert result.exists()

    # Verify output is correct size (should match input spritesheet)
    from PIL import Image
    input_img = Image.open(test_spritesheet)
    output_img = Image.open(result)
    assert output_img.size == input_img.size

    # Verify output has alpha channel
    assert output_img.mode == 'RGBA'
