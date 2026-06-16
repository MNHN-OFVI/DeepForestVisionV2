# DeepForestVision v2

DeepForestVisionv2 is an AI model designed to identify wildlife on camera trap videos and photographs from African tropical forests.

It is developed under **CC BY-NC-SA 4.0** by an academic team from the French Muséum National d'Histoire Naturelle (MNHN) as part of the One Forest Vision initiative (OFVi).

DeepForestVision v2 is also available in the AddaxAI interface (Windows / Linux / macOS, no programming required). This GitHub repository provides model weights and inference code.

**Paper:** Accepted at ICPR 2026
**Contacts:** hugo.magaldi@mnhn.fr; sabrina.krief@mnhn.fr

## What it does

Pipeline:
1) Extract frames from videos at a configurable stride
2) Detect animals / humans / vehicles with MegaDetector v5
3) Crop detections
4) Classify animal crops with a DINOv3-based classifier
5) Export:
   - `predictions_raw_*.csv` (one row per detection)
   - `predictions_stride_*_thresh_*_*.csv` (one row per media file)

Blank rule: If no detections above the detection threshold, prediction is `"blank"` and confidence is set to the **1-detection threshold**.

## Installation

### Users (run inference)
From the repo root:

```bash
pip install .
```

Or install directly from GitHub:
```bash
pip install "git+https://github.com/MNHN-OFVI/DeepForestVisionV2.git"
```
### Developers (edit code + run tests)
```bash
pip install -e ".[dev]"
pytest
```
## Run

Example:

```bash
deepforestvision \
  --data-dir ./data \
  --predictions-dir ./predictions \
  --detection-threshold 0.5 \
  --stride 1.0 \
  --device cuda:0
```
For help:

```bash
deepforestvision --help
```
## Output files

### Predictions (`predictions_stride_*_thresh_*_*.csv`)
One row per media file.

Aggregation:

- For each crop: `score[taxon] = detection_score × classifier_probability`
- Per file: mean scores across all detections
- Normalize the summed vector so taxon columns sum to 1 (per file)

Prediction:

- `Prediction` = argmax of normalized scores
- `Confidence score` = max normalized score (always between 0 and 1)

Blank files:

- `Prediction="blank"`
- Confidence score = 1-detection threshold
- all taxon columns are 0

### Raw predictions (`predictions_raw_*.csv`)

One row per detected object crop. Includes:
- source filepath + filename
- frame index + time (s) for video frames
- detection class + detection score
- crop geometry (center position, width, height)
- per-taxon scores = detection_score × classifier_probability