import argparse
import time
from typing import Optional

import cv2

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # noqa: F841
    YOLO = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLO webcam test")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to use")
    parser.add_argument("--source", type=str, default="0", help="Camera index or path")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Inference device: cpu or cuda:0"
    )
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS text")
    return parser.parse_args()


def open_capture(source_str: str, width: int, height: int) -> cv2.VideoCapture:
    if source_str.isdigit():
        source: int | str = int(source_str)
    else:
        source = source_str
    cap = cv2.VideoCapture(source)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_fps(image, fps: Optional[float]) -> None:
    if fps is None:
        return
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    if YOLO is None:
        raise RuntimeError(
            "ultralytics not available. Install with: pip install ultralytics"
        )

    model = YOLO(args.model)

    cap = open_capture(args.source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    win_name = "YOLO Realtime"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps: Optional[float] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()
            results = model.predict(
                source=frame, imgsz=max(args.width, args.height), conf=args.conf, device=args.device, verbose=False
            )
            # results[0].plot() returns annotated image (numpy array)
            annotated = results[0].plot()

            end = time.time()
            fps = 1.0 / (end - start) if (end - start) > 0 else None
            if args["show_fps"] if isinstance(args, dict) else args.show_fps:
                draw_fps(annotated, fps)

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


