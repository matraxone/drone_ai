"""Model loading utilities for YOLO and custom PyTorch models."""

from typing import Any, Optional

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore


def load_yolo(model_name: str = "yolov8n.pt") -> Optional[Any]:
    if YOLO is None:
        return None
    return YOLO(model_name)


