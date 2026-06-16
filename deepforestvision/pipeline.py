from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from supervision import ImageSink

from PytorchWildlife.models import detection as pw_detection

from .classification import load_dino_classifier, DinoClassifier
from .config import InferenceConfig
from .detection_utils import CropMeta, save_cropped_images_from_pw_results
from .io_utils import ensure_empty_dir, list_media_files, safe_copy
from .video_utils import iter_frames


DETECTION_COLUMNS_BASE = [
    "Filepath",
    "Filename",
    "Frame",
    "Time (s)",
    "Detection class",
    "Detection score",
    "center_x",
    "center_y",
    "crop_width",
    "crop_height",
]


@dataclass(frozen=True)
class Models:
    detector: object
    classifier: DinoClassifier
    taxons: List[str]        # animal labels
    taxons_all: List[str]    # animal labels + human + vehicle


def resolve_device(requested: str | None) -> str:
    """
    requested:
      - None / "" / "auto" => prefer cuda:1, then cuda:0, else cpu
      - "cuda:1" / "cuda:0" / "cpu" => use if valid, otherwise fall back to cpu
    """
    if requested is None or requested == "" or requested == "auto":
        if torch.cuda.is_available():
            # prefer cuda:1 if it exists, else cuda:0
            idx = 1 if torch.cuda.device_count() > 1 else 0
            return f"cuda:{idx}"
        return "cpu"

    # user requested something explicit
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        # If it's "cuda:N", ensure N exists
        if ":" in requested:
            try:
                idx = int(requested.split(":")[1])
                if idx >= torch.cuda.device_count():
                    return "cpu"
            except ValueError:
                return "cpu"
        return requested

    return requested
def load_models(cfg: InferenceConfig) -> Models:
    """
    Load MegaDetector and DINO classifier.
    """
    device = resolve_device(cfg.device)
    print(f"[Inference] Using device: {device}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    detector = pw_detection.MegaDetectorV5(device=device, pretrained=True)
    classifier = load_dino_classifier(
        checkpoint_path=cfg.checkpoint_path,
        dinov3_repo_dir=cfg.dinov3_repo_dir,
        device=device,
    )

    taxons = classifier.labels
    taxons_all = taxons + ["human", "vehicle"]
    return Models(detector=detector, classifier=classifier, taxons=taxons, taxons_all=taxons_all)


def make_raw_predictions_df(taxons_all: List[str]) -> pd.DataFrame:
    """
    Create the raw predictions dataframe with all expected columns.
    """
    return pd.DataFrame(columns=DETECTION_COLUMNS_BASE + taxons_all)


def _detection_geometry(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert xyxy box into:
      (center_x, center_y, width, height)
    """
    x1, y1, x2, y2 = xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_width = x2 - x1
    crop_height = y2 - y1
    return center_x, center_y, crop_width, crop_height


def _category_to_str(category: int) -> str:
    """
    MegaDetector convention:
      0 = animal
      1 = human
      2 = vehicle
    """
    if category == 1:
        return "human"
    if category == 2:
        return "vehicle"
    return "animal"


def _classify_detection(
    models: Models,
    crop_path: Path,
    category: int,
) -> List[float]:
    """
    Returns class scores aligned with models.taxons_all:
      - human: one-hot in last 2 positions
      - vehicle: one-hot in last 2 positions
      - animal: DINO probabilities for taxons, plus [0,0]
    """
    taxons_count = len(models.taxons)

    if category == 1:  # human
        return [0.0] * taxons_count + [1.0, 0.0]
    if category == 2:  # vehicle
        return [0.0] * taxons_count + [0.0, 1.0]

    # animal
    img = Image.open(crop_path).convert("RGB")
    proba = models.classifier.predict_proba(img)
    return list(proba) + [0.0, 0.0]


def detect_crop_classify_batch(
    cfg: InferenceConfig,
    models: Models,
    images_dir: Path,
    detections_dir: Path,
    file_records: List[Tuple[Path, str, str, float]],
    raw_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run detection+crop+classification for the current batch.

    file_records: list of (original_filepath, original_filename, frame_id, time_seconds)
      - For photos: frame_id="0", time_seconds=0
      - For videos: frame_id=str(extracted_frame_index)
    """
    if not file_records:
        return raw_predictions

    # Map temp frame filename -> original record
    # temp frame filename is always: "F{frame_id}_{orig_name}.JPG"
    temp_to_record: Dict[str, Tuple[Path, str, str, float]] = {}
    for orig_path, orig_name, frame_id, tsec in file_records:
        temp_name = f"F{frame_id}_{orig_name}.JPG"
        temp_to_record[temp_name] = (orig_path, orig_name, frame_id, tsec)

    # Run MegaDetector
    print("Detecting:")
    results = models.detector.batch_image_detection(
        data_path=str(images_dir),
        batch_size=32,
        det_conf_thres=cfg.detection_threshold,
    )

    # Save crops with normalized names and get metadata for each crop
    crop_index: Dict[str, CropMeta] = save_cropped_images_from_pw_results(
        results=results,
        detections_dir=detections_dir,
    )

    # Classify each crop and append to raw predictions
    for crop_filename, meta in tqdm(crop_index.items(), desc="Classifying crops", unit="crop", colour="yellow"):
        # Map back to original file using the temp frame name
        if meta.temp_frame_name not in temp_to_record:
            # Potential temp naming mismatch; safe to skip.
            continue

        orig_path, orig_name, frame_id, tsec = temp_to_record[meta.temp_frame_name]

        det_class_str = _category_to_str(meta.category)
        crop_path = detections_dir / crop_filename

        scores = _classify_detection(models, crop_path=crop_path, category=meta.category)

        # Weight class scores by detection confidence
        weighted_scores = (meta.score * np.array(scores)).tolist()

        center_x, center_y, crop_width, crop_height = _detection_geometry(meta.xyxy)

        row = [
            str(orig_path),
            orig_name,
            str(frame_id),
            float(tsec),
            det_class_str,
            float(meta.score),
            float(center_x),
            float(center_y),
            float(crop_width),
            float(crop_height),
            *weighted_scores,
        ]
        raw_predictions.loc[len(raw_predictions)] = row

    # Clear temp dirs for next batch
    ensure_empty_dir(images_dir)
    ensure_empty_dir(detections_dir)

    return raw_predictions


def consolidate_predictions(
    raw_predictions: pd.DataFrame,
    taxons_all: List[str],
) -> pd.DataFrame:
    """
    One prediction per media file by:
      1) summing scores across detections
      2) normalizing per file to get a distribution (sum=1)
      3) confidence = max(normalized_scores) in [0, 1]

    Note: raw_predictions taxon columns are expected to already be weighted by detection confidence.
    """
    cols = ["Filepath", "Filename", *taxons_all, "Prediction", "Confidence score"]

    if raw_predictions.empty:
        return pd.DataFrame(columns=cols)

    # Sum across detections (not mean)
    agg = {"Filepath": "first", "Filename": "first"}
    for t in taxons_all:
        agg[t] = "mean"

    grouped = raw_predictions.groupby("Filepath", as_index=False).agg(agg)

    score_mat = grouped[taxons_all].to_numpy(dtype=float)
    row_sums = score_mat.sum(axis=1, keepdims=True)  # (N,1)

    # Normalize safely
    norm_mat = np.zeros_like(score_mat)
    # nonblank = row_sums.squeeze(1) > 0
    norm_mat = score_mat / row_sums

    # Write normalized scores back into the dataframe (so columns sum to 1 per file)
    grouped.loc[:, taxons_all] = norm_mat

    best_idx = np.argmax(norm_mat, axis=1)
    best_val = np.max(norm_mat, axis=1)

    preds: List[str] = []
    confs: List[float] = []

    for is_nonblank, bi, bv in zip(row_sums, best_idx, best_val):
        preds.append(taxons_all[int(bi)])
        confs.append(float(bv))

    grouped["Prediction"] = preds
    grouped["Confidence score"] = confs

    return grouped[cols]


def run_inference(cfg: InferenceConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    End-to-end inference.

    Returns:
        raw_predictions, consolidated_predictions, consolidated_csv_path
    """
    cfg.predictions_dir.mkdir(parents=True, exist_ok=True)

    images_dir = cfg.temp_dir / "images"
    detections_dir = cfg.temp_dir / "detections"
    ensure_empty_dir(images_dir)
    ensure_empty_dir(detections_dir)

    models = load_models(cfg)
    raw_predictions = make_raw_predictions_df(models.taxons_all)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_files = list_media_files(cfg.data_dir, cfg.photo_extensions + cfg.video_extensions)

    # ImageSink is used for video frames (OpenCV arrays)
    sink = ImageSink(target_dir_path=str(images_dir), overwrite=False)

    batch_records: List[Tuple[Path, str, str, float]] = []
    batch_count = 0

    for path in tqdm(all_files, desc="Extracting frames", unit="file", colour="green"):
        if path.suffix in cfg.photo_extensions:
            # Photos -> just copy as a single-frame image with frame_id=0
            temp_name = f"F0_{path.name}.JPG"
            safe_copy(path, images_dir / temp_name)
            batch_records.append((path, path.name, "0", 0.0))
            batch_count += 1

        else:
            # Videos -> extract frames at stride_seconds
            for fr in iter_frames(path, stride_seconds=cfg.stride_seconds):
                temp_name = f"F{fr.index}_{path.name}.JPG"
                sink.save_image(image=fr.image_bgr, image_name=temp_name)

                batch_records.append((path, path.name, str(fr.index), float(fr.time_seconds)))
                batch_count += 1

                if batch_count >= cfg.images_max_per_batch:
                    raw_predictions = detect_crop_classify_batch(
                        cfg=cfg,
                        models=models,
                        images_dir=images_dir,
                        detections_dir=detections_dir,
                        file_records=batch_records,
                        raw_predictions=raw_predictions,
                    )
                    batch_records = []
                    batch_count = 0

    # Flush remaining batch
    raw_predictions = detect_crop_classify_batch(
        cfg=cfg,
        models=models,
        images_dir=images_dir,
        detections_dir=detections_dir,
        file_records=batch_records,
        raw_predictions=raw_predictions,
    )

    consolidated = consolidate_predictions(raw_predictions, models.taxons_all)

    # Ensure every file appears in the consolidated output (blank if no detections)
    present = set(consolidated["Filepath"].astype(str).tolist()) if not consolidated.empty else set()

    missing_rows = []
    for p in all_files:
        sp = str(p)
        if sp not in present:
            missing_rows.append(
                {
                    "Filepath": sp,
                    "Filename": p.name,
                    **{t: 0.0 for t in models.taxons_all},
                    "Prediction": "blank",
                    "Confidence score": float(1.0 - cfg.detection_threshold),
                }
            )

    if missing_rows:
        consolidated = pd.concat([consolidated, pd.DataFrame(missing_rows)], ignore_index=True)

    consolidated = consolidated.sort_values("Filepath").reset_index(drop=True)

    out_raw = cfg.predictions_dir / f"predictions_raw_{timestamp}.csv"
    out_consolidated = cfg.predictions_dir / (
        f"predictions_stride_{cfg.stride_seconds}_thresh_{cfg.detection_threshold}_{timestamp}.csv"
    )

    raw_predictions.to_csv(out_raw, index=False)
    consolidated.to_csv(out_consolidated, index=False)

    return raw_predictions, consolidated, out_consolidated
