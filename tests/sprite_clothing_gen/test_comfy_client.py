"""Tests for ComfyUI API client."""

import pytest
from sprite_clothing_gen.comfy_client import ComfyUIClient
from sprite_clothing_gen.config import COMFYUI_URL


def test_client_initialization():
    """Test client can be initialized with URL."""
    client = ComfyUIClient(COMFYUI_URL)
    assert client.base_url == COMFYUI_URL
    assert client.api_url == f"{COMFYUI_URL}/api"


def test_client_health_check():
    """Test client can check if ComfyUI is running."""
    client = ComfyUIClient(COMFYUI_URL)
    is_healthy = client.health_check()
    assert isinstance(is_healthy, bool)
    # Note: Will be True only if ComfyUI server is running
