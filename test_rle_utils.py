# test_rle_utils.py
import numpy as np
import pytest
from rle_utils import encode_rle, decode_rle

def test_encode_rle_basic():
    """Test run-length encoding compresses consecutive values."""
    mask = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)
    rle = encode_rle(mask)

    assert rle == [
        {"value": 0, "count": 3},
        {"value": 1, "count": 2},
        {"value": 0, "count": 1}
    ]

def test_decode_rle_basic():
    """Test run-length decoding reconstructs original array."""
    rle = [
        {"value": 0, "count": 3},
        {"value": 1, "count": 2},
        {"value": 0, "count": 1}
    ]
    mask = decode_rle(rle, length=6)

    expected = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal(mask, expected)

def test_encode_decode_roundtrip_2d():
    """Test 2D mask survives encode->decode roundtrip."""
    original = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0]
    ], dtype=np.uint8)

    # Flatten, encode, decode, reshape
    flat = original.flatten()
    rle = encode_rle(flat)
    decoded_flat = decode_rle(rle, length=len(flat))
    decoded_2d = decoded_flat.reshape(original.shape)

    assert np.array_equal(decoded_2d, original)

def test_decode_rle_length_mismatch():
    """Test decode_rle raises ValueError when RLE sum doesn't match expected length."""
    rle = [{"value": 0, "count": 3}, {"value": 1, "count": 2}]  # Total: 5
    with pytest.raises(ValueError, match="RLE sum .* does not match expected length"):
        decode_rle(rle, length=10)

def test_decode_rle_invalid_values():
    """Test decode_rle raises ValueError for values not 0 or 1."""
    rle = [{"value": 0, "count": 2}, {"value": 2, "count": 3}]  # Invalid value: 2
    with pytest.raises(ValueError, match="RLE values must be 0 or 1"):
        decode_rle(rle, length=5)

def test_decode_rle_negative_count():
    """Test decode_rle raises ValueError for negative counts."""
    rle = [{"value": 0, "count": -1}, {"value": 1, "count": 2}]
    with pytest.raises(ValueError, match="RLE count must be non-negative"):
        decode_rle(rle, length=1)

def test_decode_rle_missing_keys():
    """Test decode_rle raises ValueError when required keys are missing."""
    rle = [{"value": 0}]  # Missing "count"
    with pytest.raises(ValueError, match="RLE entry missing required keys"):
        decode_rle(rle, length=1)

def test_encode_rle_2d_input():
    """Test encode_rle raises ValueError for 2D array input."""
    mask_2d = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="Input mask must be 1D array"):
        encode_rle(mask_2d)
