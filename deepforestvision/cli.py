from __future__ import annotations

import argparse
from pathlib import Path

from .config import InferenceConfig
from .pipeline import run_inference


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepforestvision",
        description="DeepForestVision v2 - Camera trap photos/videos classifier for African tropical forests",
    )

    p.add_argument("--data-dir", type=Path, default=Path("./data"), help="Folder with photos and videos to process")
    p.add_argument("--predictions-dir", type=Path, default=Path("./predictions"), help="Folder where to save predictions")
    p.add_argument("--temp-dir", type=Path, default=Path("./temp"), help="Temporary folder for extracted frames and crops")

    p.add_argument("--detection-threshold", type=float, default=0.5, help="MegaDetector score threshold")
    p.add_argument("--stride", type=float, default=1.0, help="Video frame extraction stride (seconds)")
    p.add_argument("--images-max", type=int, default=3000, help="Max frames/photos per detection batch")

    p.add_argument("--device", type=str, default=None, help='Device like "cuda", "cuda:0", or "cpu"')
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = InferenceConfig(
        data_dir=args.data_dir,
        predictions_dir=args.predictions_dir,
        temp_dir=args.temp_dir,
        detection_threshold=args.detection_threshold,
        stride_seconds=args.stride,
        images_max_per_batch=args.images_max,
        device=args.device,
    )
    _, _, out_path = run_inference(cfg)
    print(f"✅ Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()