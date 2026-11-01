# SponsorVision

SponsorVision is a local computer vision system that measures on-screen exposure of Collingwood Football Club sponsor logos in AFL broadcast footage. It detects seven sponsor logos with a YOLO11n detector, converts the detections into exposure estimates per sponsor, per quarter and per match, and presents the results in a Streamlit dashboard.

All footage and derived data stay on the local machine. Do not upload footage, frame extracts, datasets or outputs to a public repository or external service.

## Requirements

- Windows 11 (the scripts use Windows paths)
- Python 3.10 or 3.11
- An NVIDIA GPU with CUDA support for the export script (see Known issues)

## Setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

Open the address Streamlit prints, usually http://localhost:8501. Select a video, set the sampling interval (default 1.0 second) and confidence threshold (default 0.35), then click **Analyse video now**. Results are saved to `output\sponsorvision_reports\`, with a CSV of detections, a summary CSV, a metadata JSON and a ZIP of the run.

## Export detections from the command line

```powershell
python export_sponsor_csv.py --video data\videos\<match>.mp4 --fps 5 --conf 0.25
```

Frames are sampled at `--fps` (default 5, so each sampled frame represents 0.2 seconds). Results are written to `output\sponsor_csv\`.

## Evaluate the detector

```powershell
yolo detect val model=runs\detect\runs\detect\sponsorvision_v7-2\weights\best.pt data=datasets\sponsorvision_v7\data.yaml split=test
```

## Model and dataset

- **Trained weights:** `runs\detect\runs\detect\sponsorvision_v7-2\weights\best.pt` (the default model in the dashboard)
- **Dataset:** `datasets\sponsorvision_v7\` with 1,497 training, 450 validation and 367 test images
- **Classes:** kfc, emirates, nike, la trobe, primo, rayw, kgm

### Results on the held-out test split

| Metric | Value |
|---|---|
| Precision | 0.43 |
| Recall | 0.50 |
| mAP50 | 0.44 |
| mAP50-95 | 0.18 |

Per-class results are in the `runs\detect\val` folder after running the evaluation command. Emirates and rayw have no instances in the test split.

## How exposure is calculated

Each sampled frame in which a sponsor is detected adds one sampling interval to that sponsor's exposure. A sponsor receives at most one count per frame, however many boxes are found. The result is an estimate of on-screen time, not a record of continuous exposure.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `sponsor_reporting.py` | Saves analysis runs, CSVs, metadata and ZIP files |
| `export_sponsor_csv.py` | Command-line detection export |
| `requirements.txt` | Python dependencies |
| `datasets\sponsorvision_v7\` | Training, validation and test data |
| `runs\detect\runs\detect\sponsorvision_v7-2\` | Trained weights, training logs and curves |
| `output\` | Analysis outputs (not for version control) |
| `data\videos\` | Source footage (not for version control) |

## Known issues and limitations

- **Hard-coded paths:** `datasets\sponsorvision_v5\data.yaml`, `datasets\sponsorvision_v6\data.yaml`, `sponsor_dataset_v3\` and `sponsor_dataset_v4\` contain absolute paths from other machines. Update the `path:` line before using them. The v7 `data.yaml` should use a path that exists on the current machine.
- **GPU requirement:** `export_sponsor_csv.py` runs the model on GPU (`device=0`). On a CPU-only machine, change this to `device="cpu"`.
- **No object tracking:** the pipeline does not follow logos from frame to frame, so it cannot count distinct appearances.
- **Sampling and threshold:** exposure depends on both the sampling interval and the confidence threshold. Choose the threshold from the precision-recall curves.
- **Legacy reference matching:** the ORB reference-image option in the dashboard is an earlier prototype feature. It is not used for the reported results.

## Confidentiality

Footage, frame extracts, datasets and outputs are confidential. Keep them on approved local or private infrastructure. Add `data\`, `output\`, `datasets\`, `training_sources\` and `runs\` to `.gitignore` before any code is shared.
