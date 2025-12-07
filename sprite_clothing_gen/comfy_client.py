"""ComfyUI API client for workflow execution."""

import json
import uuid
import requests
from typing import Dict, Any, Optional
from pathlib import Path


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(self, base_url: str):
        """Initialize client with ComfyUI server URL.

        Args:
            base_url: Base URL of ComfyUI server (e.g., http://127.0.0.1:8188)
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if ComfyUI server is running.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow dictionary

        Returns:
            Prompt ID for tracking execution

        Raises:
            RuntimeError: If prompt queueing fails
        """
        prompt_id = str(uuid.uuid4())
        payload = {
            "prompt": workflow,
            "client_id": prompt_id
        }

        response = self.session.post(
            f"{self.api_url}/prompt",
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to queue prompt: {response.text}")

        result = response.json()
        return result.get("prompt_id", prompt_id)

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get execution history for a prompt.

        Args:
            prompt_id: ID of the prompt to check

        Returns:
            History dict if available, None otherwise
        """
        response = self.session.get(f"{self.api_url}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            return history.get(prompt_id)
        return None

    def upload_image(self, image_path: Path, subfolder: str = "") -> str:
        """Upload an image to ComfyUI input directory.

        Args:
            image_path: Path to image file
            subfolder: Optional subfolder in input directory

        Returns:
            Filename of uploaded image

        Raises:
            RuntimeError: If upload fails
        """
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/png')}
            data = {'subfolder': subfolder, 'overwrite': 'true'}

            response = self.session.post(
                f"{self.base_url}/upload/image",
                files=files,
                data=data
            )

            if response.status_code != 200:
                raise RuntimeError(f"Failed to upload image: {response.text}")

            result = response.json()
            return result['name']

    def download_image(self, filename: str, subfolder: str = "", output_dir: Optional[Path] = None) -> Path:
        """Download an image from ComfyUI output directory.

        Args:
            filename: Name of the image file
            subfolder: Optional subfolder in output directory
            output_dir: Directory to save image (defaults to current dir)

        Returns:
            Path to downloaded image

        Raises:
            RuntimeError: If download fails
        """
        params = {'filename': filename, 'subfolder': subfolder, 'type': 'output'}
        response = self.session.get(f"{self.base_url}/view", params=params)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to download image: {response.text}")

        if output_dir is None:
            output_dir = Path.cwd()

        output_path = output_dir / filename
        with open(output_path, 'wb') as f:
            f.write(response.content)

        return output_path
