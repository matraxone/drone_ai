"""Real-time object detection using YOLOv8/YOLOv5 via ultralytics or PyTorch hub."""

from typing import Any, List

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # Fallback placeholder if ultralytics not present on dev host
    YOLO = None  # type: ignore


class Detector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model = YOLO(model_name) if YOLO else None

    def predict(self, frame) -> List[Any]:  # Replace Any with a concrete type later
        if not self.model:
            return []
        results = self.model(frame, verbose=False)
        return results


