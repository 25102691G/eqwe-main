"""Helpers for decoding uploaded image bytes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray | None:
    """Decode raw image bytes into an OpenCV BGR image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_image_file_to_bgr(image_path: str | Path) -> np.ndarray | None:
    """Decode one local image file into an OpenCV BGR image."""
    path = Path(image_path)
    if not path.exists():
        return None
    return decode_image_bytes_to_bgr(path.read_bytes())


__all__ = ["decode_image_bytes_to_bgr", "decode_image_file_to_bgr"]
