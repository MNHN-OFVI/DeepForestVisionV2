from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropMeta:
    """
    Metadata for a single saved detection crop.
    """
    temp_frame_name: str  # e.g. "F30_SEB_184454.AVI.JPG"
    det_idx: int
    category: int         # 0 animal, 1 human, 2 vehicle
    score: float
    xyxy: Tuple[float, float, float, float]  # pixel coords (x1,y1,x2,y2)


# crop__<temp_frame_name>__det<000>.jpg
_CROP_RE = re.compile(r"^crop__(?P<temp>.+)__det(?P<det>\d{3})\.jpg$")


def parse_crop_filename(crop_filename: str) -> Tuple[str, int]:
    """
    Parse normalized crop filename:
      crop__F30_SEB_184454.AVI.JPG__det002.jpg -> ("F30_SEB_184454.AVI.JPG", 2)
    """
    m = _CROP_RE.match(crop_filename)
    if not m:
        raise ValueError(f"Unrecognized normalized crop filename: {crop_filename}")
    return m.group("temp"), int(m.group("det"))


def _sanitize_for_filename(name: str) -> str:
    """
    Keep filenames safe across OSes while staying readable.
    Allows letters, numbers, dot, underscore, dash.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _iter_supervision_detections(res: Dict[str, Any]) -> Iterator[Tuple[int, int, float, Tuple[float, float, float, float]]]:
    """
    Yield (det_idx, category, score, xyxy_pixels) from a MegaDetector result dict.

    Expected structure:
      res["detections"] is supervision.Detections with:
        - .xyxy (N,4) float
        - .confidence (N,) float
        - .class_id (N,) int
    """
    dets = res.get("detections", None)
    if dets is None:
        return

    xyxy = np.asarray(dets.xyxy)          # (N,4)
    conf = np.asarray(dets.confidence)    # (N,)
    cls = np.asarray(dets.class_id)       # (N,)

    n = xyxy.shape[0]
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i].tolist()
        yield i, int(cls[i]), float(conf[i]), (float(x1), float(y1), float(x2), float(y2))


def save_cropped_images_from_pw_results(
    results: list[dict],
    detections_dir: Path,
) -> Dict[str, CropMeta]:
    """
    Save detection crops using normalized crop naming and return metadata indexed by crop filename.

    Args:
        results: output of MegaDetectorV5.batch_image_detection(...) -> list[dict]
        detections_dir: directory where crops are saved

    Returns:
        Dict[crop_filename -> CropMeta]
    """
    detections_dir.mkdir(parents=True, exist_ok=True)
    crop_index: Dict[str, CropMeta] = {}

    for res in results:
        img_id = res.get("img_id")
        if not img_id:
            continue

        img_path = Path(img_id)
        temp_frame_name = img_path.name  # "F30_SEB_184454.AVI.JPG"
        safe_temp = _sanitize_for_filename(temp_frame_name)

        # Open once per frame
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        for det_idx, category, score, (x1, y1, x2, y2) in _iter_supervision_detections(res):
            # Clip to image bounds (prevents PIL errors / empty crops)
            x1 = max(0.0, min(x1, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            x2 = max(1.0, min(x2, float(w)))
            y2 = max(1.0, min(y2, float(h)))

            crop = img.crop((x1, y1, x2, y2))

            crop_filename = f"crop__{safe_temp}__det{det_idx:03d}.jpg"
            crop_path = detections_dir / crop_filename
            crop.save(crop_path, quality=95)

            crop_index[crop_filename] = CropMeta(
                temp_frame_name=temp_frame_name,
                det_idx=det_idx,
                category=category,
                score=score,
                xyxy=(x1, y1, x2, y2),
            )

    return crop_index