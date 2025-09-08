"""Camera abstraction for Raspberry Pi camera or USB cameras."""

from typing import Iterator, Optional
import cv2


class VideoStream:
    def __init__(self, source: int | str = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.cap = cv2.VideoCapture(source)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def frames(self) -> Iterator[tuple[bool, "cv2.Mat"]]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield ok, frame

    def release(self) -> None:
        self.cap.release()


