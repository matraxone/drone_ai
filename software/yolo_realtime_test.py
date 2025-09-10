import argparse
import time
import threading
import queue
import cv2
from typing import Optional
from collections import deque

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLO webcam test (optimized)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to use")
    parser.add_argument("--source", type=str, default="0", help="Camera index or path")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=360, help="Capture height")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: cpu or cuda:0")
    parser.add_argument("--max-det", type=int, default=30, help="Max detections per frame")
    parser.add_argument("--imgsz", type=int, default=256, help="Inference image size (smaller = faster)")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA")
    # show FPS by default so it's visible at startup; use --no-show-fps to disable
    parser.add_argument("--show-fps", action="store_true", default=True, help="Show FPS (default: on)")
    return parser.parse_args()


def open_capture(source_str: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source_str) if source_str.isdigit() else source_str)
    # try to reduce camera buffer to minimize latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_fps(image, fps: Optional[float]) -> None:
    # always draw something so user sees the FPS box at startup
    text = f"FPS: {fps:.1f}" if (fps is not None) else "FPS: --"
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    margin = 8
    pad = 6

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # compute coordinates for bottom-right box
    x = max(0, w - margin - text_w)
    y = max(0, h - margin)

    x1 = max(0, x - pad)
    y1 = max(0, y - text_h - pad - baseline)
    x2 = min(w, x + text_w + pad)
    y2 = min(h, y + pad)

    # draw background rectangle and text
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    text_x = x1 + pad
    text_y = y2 - pad - baseline
    text_y = max(text_h, text_y)
    cv2.putText(image, text, (text_x, text_y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def _capture_thread(cap: cv2.VideoCapture, q: "queue.Queue[Optional]" , stop_evt: threading.Event):
    """Continuously read frames and keep only the latest in the queue."""
    while not stop_evt.is_set():
        ok, frame = cap.read()
        if not ok:
            # signal termination
            try:
                q.put(None, block=False)
            except Exception:
                pass
            stop_evt.set()
            break
        # keep only most recent frame
        try:
            q.put(frame, block=False)
        except queue.Full:
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put(frame, block=False)
            except Exception:
                pass
    try:
        q.put(None, block=False)
    except Exception:
        pass


# nuovo: media mobile per FPS (accurata e stabile)
class FPSMeter:
    def __init__(self, window: int = 16):
        self.times = deque(maxlen=window)

    def update(self, t: float) -> None:
        self.times.append(t)

    def get_fps(self) -> Optional[float]:
        if len(self.times) < 2:
            return None
        # fps = frames / elapsed_time
        elapsed = self.times[-1] - self.times[0]
        if elapsed <= 0:
            return None
        return len(self.times) / elapsed


def main() -> None:
    args = parse_args()
    if YOLO is None:
        raise RuntimeError("Install ultralytics with: pip install ultralytics")

    model = YOLO(args.model)

    # move model to device once
    try:
        if hasattr(model, "to"):
            model.to(args.device)
    except Exception:
        pass

    # enable half precision when requested and CUDA available
    if args.half and "cuda" in args.device:
        try:
            if hasattr(model, "model") and hasattr(model.model, "half"):
                model.model.half()
        except Exception:
            pass

    cap = open_capture(args.source, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    frame_q: "queue.Queue" = queue.Queue(maxsize=1)
    stop_evt = threading.Event()
    threading.Thread(target=_capture_thread, args=(cap, frame_q, stop_evt), daemon=True).start()

    win_name = "YOLO Realtime"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # start with None so draw_fps shows "FPS: --" at program start
    fps = None
    fps_meter = FPSMeter(window=16)
    try:
        while True:
            try:
                frame = frame_q.get(timeout=1.0)
            except queue.Empty:
                break
            if frame is None:
                break

            # show current FPS immediately (before inference) so user sees it at startup
            if args.show_fps:
                pre_img = frame.copy()
                draw_fps(pre_img, fps)
                cv2.imshow(win_name, pre_img)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            start = time.time()
            # lightweight inference: small imgsz, only persons, limited detections
            try:
                results = model(frame, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, classes=[0], max_det=args.max_det, verbose=False)
            except Exception:
                # fallback older API
                results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                        device=args.device, classes=[0], max_det=args.max_det, verbose=False)

            annotated = frame  # annotate in-place to avoid extra copy
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                    except Exception:
                        try:
                            xyxy = boxes.xyxy.numpy()
                            confs = boxes.conf.numpy()
                        except Exception:
                            xyxy = []
                            confs = []

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        conf = float(confs[i]) if i < len(confs) else 0.0
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, f"person {conf:.2f}", (x1, max(y1 - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            end = time.time()
            # aggiorna FPS meter usando timestamp di fine elaborazione
            fps_meter.update(end)
            fps = fps_meter.get_fps()
            if args.show_fps:
                draw_fps(annotated, fps)

            cv2.imshow(win_name, annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        stop_evt.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
