# test_rle_utils.py
import numpy as np
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
