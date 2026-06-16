from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


PHOTO_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
VIDEO_EXTENSIONS: Tuple[str, ...] = (".avi", ".mov", ".mp4", ".AVI", ".MOV", ".MP4")


@dataclass(frozen=True)
class InferenceConfig:
    # Paths
    data_dir: Path
    predictions_dir: Path
    temp_dir: Path = Path("./temp")

    # Detection / video settings
    detection_threshold: float = 0.5
    stride_seconds: float = 1.0  # seconds between extracted frames
    images_max_per_batch: int = 3000

    # Compute
    device: str = "cuda"  # e.g. "cuda", "cuda:0", "cpu"

    # DINO classifier checkpointing
    checkpoint_path: Path = Path("./weights/DeepForestVisionV2.pth")
    dinov3_repo_dir: Path = Path("./weights/dinov3")

    # I/O
    photo_extensions: Tuple[str, ...] = PHOTO_EXTENSIONS
    video_extensions: Tuple[str, ...] = VIDEO_EXTENSIONS