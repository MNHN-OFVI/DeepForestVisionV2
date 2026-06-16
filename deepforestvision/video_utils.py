from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
from supervision.utils import video as supervision_video


@dataclass(frozen=True)
class ExtractedFrame:
    """
    A frame extracted from a video.
    """
    index: int          # 0..N-1 extracted frames index (not original frame number)
    time_seconds: float # index * stride_seconds
    image_bgr: "object" # numpy array (OpenCV BGR)


def get_video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0
    finally:
        cap.release()


def iter_frames(path: Path, stride_seconds: float) -> Generator[ExtractedFrame, None, None]:
    """
    Yield frames at roughly every `stride_seconds`.

    We use supervision's generator with stride in *frames*:
      stride_frames = int(stride_seconds * fps)

    If fps can't be read, default stride_frames=1.
    """
    fps = get_video_fps(path)
    stride_frames = max(1, int(stride_seconds * fps)) if fps > 0 else 1

    idx = 0
    for frame in supervision_video.get_video_frames_generator(
        source_path=str(path),
        stride=stride_frames,
    ):
        yield ExtractedFrame(index=idx, time_seconds=idx * stride_seconds, image_bgr=frame)
        idx += 1