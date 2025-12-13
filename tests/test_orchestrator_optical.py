import pytest
from pathlib import Path
from PIL import Image
import tempfile


class TestSpriteClothingGeneratorOptical:
    def test_generate_creates_output_spritesheet(self, tmp_path):
        """Generate should create output spritesheet from inputs."""
        from sprite_clothing_gen.orchestrator_optical import SpriteClothingGenerator

        # Create 2x2 base spritesheet (simplified for test)
        base = Image.new('RGB', (100, 100), color=(255, 255, 255))
        # Add gray mannequin in each quadrant
        for qx, qy in [(0, 0), (50, 0), (0, 50), (50, 50)]:
            for x in range(qx + 15, qx + 35):
                for y in range(qy + 15, qy + 35):
                    base.putpixel((x, y), (128, 128, 128))
        base_path = tmp_path / "base.png"
        base.save(base_path)

        # Create 2x2 clothed spritesheet
        clothed = Image.new('RGB', (100, 100), color=(255, 255, 255))
        for qx, qy in [(0, 0), (50, 0), (0, 50), (50, 50)]:
            for x in range(qx + 15, qx + 35):
                for y in range(qy + 15, qy + 35):
                    clothed.putpixel((x, y), (100, 80, 60))  # Brown armor
        clothed_path = tmp_path / "clothed.png"
        clothed.save(clothed_path)

        output_path = tmp_path / "output.png"

        generator = SpriteClothingGenerator(temp_dir=tmp_path / "temp")
        result = generator.generate(
            base_spritesheet=base_path,
            clothed_spritesheet=clothed_path,
            output_path=output_path,
            grid_size=(2, 2)
        )

        assert result == output_path
        assert output_path.exists()

        # Verify output is correct size
        output_img = Image.open(output_path)
        assert output_img.size == (100, 100)
