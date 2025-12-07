"""Main orchestrator for sprite clothing generation pipeline."""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from sprite_clothing_gen.comfy_client import ComfyUIClient
from sprite_clothing_gen.spritesheet_utils import split_spritesheet, reassemble_spritesheet
from sprite_clothing_gen.clothing_extractor import extract_clothing_from_reference
from sprite_clothing_gen.workflow_builder import (
    build_openpose_preprocessing_workflow,
    build_clothing_generation_workflow
)
from sprite_clothing_gen.config import (
    COMFYUI_URL,
    GRID_SIZE,
    FRAME_COUNT,
    TEMP_DIR,
    U2NET_MODEL
)


class SpriteClothingGenerator:
    """Orchestrates the sprite clothing generation pipeline."""

    def __init__(self, comfyui_url: str = COMFYUI_URL):
        """Initialize generator with ComfyUI client.

        Args:
            comfyui_url: URL of ComfyUI server
        """
        self.client = ComfyUIClient(comfyui_url)
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        base_spritesheet: Path,
        reference_frame: Path,
        reference_frame_index: int,
        output_path: Path,
        seed: int = 42,
        keep_temp: bool = False
    ) -> Path:
        """Generate clothing-only spritesheet from base spritesheet and reference.

        Args:
            base_spritesheet: Path to base 5x5 spritesheet
            reference_frame: Path to single clothed reference frame
            reference_frame_index: Which frame index the reference corresponds to (0-24)
            output_path: Path to save output clothing spritesheet
            seed: Random seed for consistent generation
            keep_temp: If True, keep temporary files

        Returns:
            Path to output clothing spritesheet

        Raises:
            RuntimeError: If generation fails
            ValueError: If reference_frame_index is out of range
        """
        if not 0 <= reference_frame_index < FRAME_COUNT:
            raise ValueError(
                f"reference_frame_index must be 0-{FRAME_COUNT-1}, got {reference_frame_index}"
            )

        print(f"🎨 Starting sprite clothing generation...")
        print(f"   Base spritesheet: {base_spritesheet}")
        print(f"   Reference frame: {reference_frame}")
        print(f"   Reference frame index: {reference_frame_index}")

        # Check ComfyUI is running
        if not self.client.health_check():
            raise RuntimeError(
                f"ComfyUI server not accessible at {self.client.base_url}. "
                "Please start ComfyUI first."
            )

        try:
            # Step 1: Split base spritesheet
            print("\n📦 Step 1: Splitting base spritesheet...")
            frames_dir = self.temp_dir / "frames"
            base_frames = split_spritesheet(base_spritesheet, frames_dir, GRID_SIZE)
            print(f"   Split into {len(base_frames)} frames")

            # Step 2: Extract clothing from reference
            print("\n✂️  Step 2: Extracting clothing from reference...")
            clothing_ref_path = self.temp_dir / "reference_clothing_only.png"
            extract_clothing_from_reference(
                reference_frame,
                clothing_ref_path,
                model=U2NET_MODEL
            )
            print(f"   Saved clothing-only reference to {clothing_ref_path.name}")

            # Step 3: Generate OpenPose skeletons for all frames
            print("\n🦴 Step 3: Generating OpenPose skeletons...")
            pose_frames = self._generate_pose_skeletons(base_frames)
            print(f"   Generated {len(pose_frames)} pose skeletons")

            # Step 4: Generate clothing for each frame
            print("\n🎨 Step 4: Generating clothing layers...")
            clothing_frames = self._generate_clothing_layers(
                pose_frames,
                clothing_ref_path,
                seed
            )
            print(f"   Generated {len(clothing_frames)} clothing layers")

            # Step 5: Reassemble into spritesheet
            print("\n🔧 Step 5: Reassembling spritesheet...")
            result = reassemble_spritesheet(clothing_frames, output_path, GRID_SIZE)
            print(f"   Saved output to {output_path}")

            # Cleanup
            if not keep_temp:
                print("\n🧹 Cleaning up temporary files...")
                self._cleanup_temp_files()

            print("\n✅ Generation complete!")
            return result

        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            raise

    def _generate_pose_skeletons(self, frame_paths: List[Path]) -> List[Path]:
        """Generate OpenPose skeletons for all frames.

        Args:
            frame_paths: List of paths to base frames

        Returns:
            List of paths to pose skeleton images
        """
        pose_dir = self.temp_dir / "poses"
        pose_dir.mkdir(exist_ok=True)
        pose_frames = []

        for i, frame_path in enumerate(frame_paths):
            # Upload frame to ComfyUI
            filename = self.client.upload_image(frame_path, subfolder="")

            # Build and queue workflow
            workflow = build_openpose_preprocessing_workflow(
                filename,
                output_filename_prefix=f"pose_{i:02d}"
            )
            prompt_id = self.client.queue_prompt(workflow)

            # Wait for completion and get output filename
            history = self._wait_for_completion(prompt_id)

            # Extract output filename from history
            output_filename = self._extract_output_filename(history)

            # Download result
            pose_path = pose_dir / f"pose_{i:02d}.png"
            self.client.download_image(output_filename, output_dir=pose_dir)

            # Rename to standard name
            downloaded = pose_dir / output_filename
            if downloaded.exists():
                downloaded.rename(pose_path)

            pose_frames.append(pose_path)
            print(f"   Generated pose {i+1}/{len(frame_paths)}")

        return pose_frames

    def _generate_clothing_layers(
        self,
        pose_frames: List[Path],
        clothing_reference: Path,
        seed: int
    ) -> List[Path]:
        """Generate clothing layers for all frames.

        Args:
            pose_frames: List of paths to pose skeleton images
            clothing_reference: Path to clothing-only reference image
            seed: Random seed for generation

        Returns:
            List of paths to generated clothing layer images
        """
        clothing_dir = self.temp_dir / "clothing"
        clothing_dir.mkdir(exist_ok=True)
        clothing_frames = []

        # Upload clothing reference once
        ref_filename = self.client.upload_image(clothing_reference, subfolder="")

        for i, pose_path in enumerate(pose_frames):
            # Upload pose skeleton
            pose_filename = self.client.upload_image(pose_path, subfolder="")

            # Build and queue workflow
            workflow = build_clothing_generation_workflow(
                pose_filename,
                ref_filename,
                seed=seed,
                output_filename_prefix=f"clothing_{i:02d}"
            )
            prompt_id = self.client.queue_prompt(workflow)

            # Wait for completion and get output filename
            history = self._wait_for_completion(prompt_id)

            # Extract output filename from history
            output_filename = self._extract_output_filename(history)

            # Download result
            clothing_path = clothing_dir / f"clothing_{i:02d}.png"
            self.client.download_image(output_filename, output_dir=clothing_dir)

            # Rename to standard name
            downloaded = clothing_dir / output_filename
            if downloaded.exists():
                downloaded.rename(clothing_path)

            clothing_frames.append(clothing_path)
            print(f"   Generated clothing layer {i+1}/{len(pose_frames)}")

        return clothing_frames

    def _wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for a prompt to complete execution.

        Args:
            prompt_id: ID of the prompt to wait for
            timeout: Maximum seconds to wait

        Returns:
            History dict containing outputs

        Raises:
            RuntimeError: If execution times out or fails
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Execution timed out after {timeout}s")

            history = self.client.get_history(prompt_id)
            if history is not None:
                # Check status
                status = history.get("status", {})
                status_str = status.get("status_str", "")

                # Check if execution failed
                if status_str == "error":
                    messages = status.get("messages", [])
                    error_msg = "Unknown error"
                    for msg in messages:
                        if msg[0] == "execution_error":
                            error_msg = str(msg[1])
                            break
                    raise RuntimeError(f"Workflow execution failed: {error_msg}")

                # Check if execution completed successfully
                if "outputs" in history and history["outputs"]:
                    return history

            time.sleep(1)

    def _extract_output_filename(self, history: Dict[str, Any]) -> str:
        """Extract output filename from ComfyUI history.

        Args:
            history: History dict from ComfyUI

        Returns:
            Filename of the generated image

        Raises:
            RuntimeError: If filename cannot be extracted
        """
        try:
            # Navigate history structure to find the SaveImage node output
            outputs = history.get("outputs", {})
            # Find the first SaveImage node (typically last node in workflow)
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    images = node_output["images"]
                    if images and len(images) > 0:
                        return images[0]["filename"]

            raise RuntimeError("No output image found in history")
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Failed to extract filename from history: {e}")

    def _cleanup_temp_files(self) -> None:
        """Remove temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
