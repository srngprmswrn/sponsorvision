"""Persistent CSV and ZIP reporting for SponsorVision match analyses.

All times are estimates based on video sampling, not continuous tracking.
A sponsor contributes at most one exposure duration per sampled frame,
regardless of how many bounding boxes are detected for that sponsor.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "output" / "sponsorvision_reports"

DETECTION_COLUMNS = [
    "video", "quarter", "sample_index", "timestamp_sec", "sponsor",
    "x1", "y1", "x2", "y2", "method", "score", "inlier_matches",
]
INTERVAL_COLUMNS = [
    "video", "quarter", "sponsor", "start_sec", "end_sec",
    "duration_sec", "detected_samples",
]
SUMMARY_COLUMNS = [
    "video", "quarter", "sponsor", "detected_samples", "total_boxes",
    "sampled_visible_seconds", "sampled_visible_minutes",
    "appearance_runs_estimate", "percent_decoded_samples", "percent_video_estimate",
]


def generate_reports(
    detections: pd.DataFrame,
    presence: pd.DataFrame,
    sponsors: list[str],
    video_name: str,
    quarter: int,
    video_duration: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate consistent interval and sponsor summary tables.

    An interval is a run of consecutive *sampled* timestamps where the logo
    was detected. One missed sample ends a run. No detection smoothing is used.
    """
    records: list[dict] = []
    intervals: list[dict] = []
    decoded_count = int(presence["decoded"].sum()) if not presence.empty else 0
    boxes = detections["sponsor"].value_counts().to_dict() if not detections.empty else {}

    for sponsor in sponsors:
        detected_samples = 0
        exposure_seconds = 0.0
        active = None
        previous_index = None
        last_end = None
        run_samples = 0
        runs = 0

        def finish_interval():
            nonlocal active, previous_index, last_end, run_samples, runs
            if active is not None:
                intervals.append({
                    "video": video_name,
                    "quarter": quarter,
                    "sponsor": sponsor,
                    "start_sec": round(active, 3),
                    "end_sec": round(last_end, 3),
                    "duration_sec": round(last_end - active, 3),
                    "detected_samples": run_samples,
                })
                runs += 1
            active = None
            previous_index = None
            last_end = None
            run_samples = 0

        for data in presence.to_dict(orient="records"):
            current_index = int(data["sample_index"])
            has_sponsor = bool(data["decoded"]) and bool(data[sponsor])
            if not has_sponsor:
                finish_interval()
                continue
            timestamp = float(data["timestamp_sec"])
            sample_duration = float(data["sample_duration_sec"])
            if active is None or previous_index is None or current_index != previous_index + 1:
                finish_interval()
                active = timestamp
            detected_samples += 1
            exposure_seconds += sample_duration
            previous_index = current_index
            last_end = timestamp + sample_duration
            run_samples += 1
        finish_interval()

        records.append({
            "video": video_name,
            "quarter": quarter,
            "sponsor": sponsor,
            "detected_samples": detected_samples,
            "total_boxes": int(boxes.get(sponsor, 0)),
            "sampled_visible_seconds": round(exposure_seconds, 3),
            "sampled_visible_minutes": round(exposure_seconds / 60.0, 4),
            "appearance_runs_estimate": runs,
            "percent_decoded_samples": round(100.0 * detected_samples / decoded_count, 2) if decoded_count else 0.0,
            "percent_video_estimate": round(100.0 * exposure_seconds / video_duration, 2) if video_duration > 0 else 0.0,
        })

    interval_df = pd.DataFrame(intervals, columns=INTERVAL_COLUMNS)
    if not interval_df.empty:
        interval_df = interval_df.sort_values(["start_sec", "sponsor"], kind="stable").reset_index(drop=True)
    summary_df = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    return interval_df, summary_df


def save_analysis(
    detections: pd.DataFrame,
    presence: pd.DataFrame,
    intervals: pd.DataFrame,
    summary: pd.DataFrame,
    video_name: str,
    model_path: str,
    quarter: int,
    sample_step: float,
    confidence: float,
    duration: float,
    decoded: int,
    attempted: int,
    reference_names: list[str],
) -> Path:
    """Save an immutable per-video run with 4 CSV files, metadata and a ZIP."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(video_name).stem).strip("._-")[:55] or "video"
    folder = REPORTS_DIR / f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_stem}"
    folder.mkdir(parents=False, exist_ok=False)

    files = {
        "detections.csv": detections,
        "frame_presence.csv": presence,
        "exposure_intervals.csv": intervals,
        "exposure_summary.csv": summary,
    }
    for filename, frame in files.items():
        frame.to_csv(folder / filename, index=False, encoding="utf-8")

    metadata = {
        "video": video_name,
        "quarter": int(quarter),
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "sample_every_seconds": float(sample_step),
        "confidence_threshold": float(confidence),
        "video_duration_seconds": round(float(duration), 3),
        "decoded_samples": int(decoded),
        "attempted_samples": int(attempted),
        "detected_boxes": int(len(detections)),
        "sponsor_classes": summary["sponsor"].tolist(),
        "orb_reference_images": list(reference_names),
        "methodology": "Sampling-based visibility approximation. One logo class counted once per sample; missing samples break intervals. No gap interpolation or ground-truth verification.",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with ZipFile(folder / "all_csv_reports.zip", "w", ZIP_DEFLATED) as archive:
        for filename in (*files.keys(), "metadata.json"):
            archive.write(folder / filename, arcname=filename)
    return folder


def list_analyses() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(
        (p for p in REPORTS_DIR.iterdir() if p.is_dir() and (p / "metadata.json").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )


def load_analysis(folder: Path) -> dict:
    folder = Path(folder)
    if folder.parent.resolve() != REPORTS_DIR.resolve():
        raise ValueError("Reports must be loaded from the SponsorVision reports directory")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return {
        "folder": folder,
        "metadata": metadata,
        "detections": pd.read_csv(folder / "detections.csv"),
        "presence": pd.read_csv(folder / "frame_presence.csv"),
        "intervals": pd.read_csv(folder / "exposure_intervals.csv"),
        "summary": pd.read_csv(folder / "exposure_summary.csv"),
    }
