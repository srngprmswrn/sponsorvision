"""SponsorVision | Collingwood-inspired sponsor intelligence dashboard."""
import math
import tempfile
from pathlib import Path

from sponsor_reporting import (
    DETECTION_COLUMNS, generate_reports, save_analysis,
    list_analyses, load_analysis,
)

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SponsorVision | Collingwood",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0b0b0d; color: #f6f6f6; }
[data-testid="stSidebar"] { background: #151517; border-right: 1px solid #343438; }
[data-testid="stHeader"] { background: #0b0b0d; }
.block-container { padding-top: 2rem; max-width: 1450px; }
h1, h2, h3 { letter-spacing: -.035em; }
div[data-testid="stMetric"] {
  background: #19191c; border: 1px solid #36363a;
  border-radius: 14px; padding: 17px 19px;
}
div[data-testid="stMetricValue"] { color: white; }
.stButton > button[kind="primary"] {
  background: white; color: #111; border: 1px solid white; font-weight: 800;
}
.stButton > button[kind="primary"]:hover {
  background: #dddddd; color: #111; border-color: #dddddd;
}
.hero {
  padding: 32px 35px; border: 1px solid #424247; border-radius: 18px;
  background: linear-gradient(120deg,#171719,#101012);
  margin-bottom: 12px;
}
.eyebrow { font-size: 12px; font-weight: 800; letter-spacing: .2em; color: #c9c9cd; }
.hero-title { font-size: clamp(32px, 5vw, 58px); font-weight: 900; letter-spacing: -.055em; margin: 10px 0; }
.hero-sub { color: #d2d2d6; max-width: 720px; font-size: 16px; }
.small-note { color: #aaaab1; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# Place a club logo image at assets/collingwood_logo.png to display it in the banner.
logo_path = Path(__file__).resolve().parent / "assets" / "collingwood_logo.jpeg"
brand_left, brand_right = st.columns([1, 6], vertical_alignment="center")
with brand_left:
    if logo_path.is_file():
        st.image(str(logo_path), width=112)
    else:
        st.markdown(
            '<div style="border:2px solid #ddd;border-radius:12px;text-align:center;'
            'padding:19px 5px;font-size:27px;font-weight:900;">CFC</div>',
            unsafe_allow_html=True,
        )
with brand_right:
    st.markdown("## SPONSORVISION")
    st.caption("COLLINGWOOD FC  /  SPONSOR INTELLIGENCE")

st.markdown(
    '<div style="height:4px;background:linear-gradient(90deg,#eee 0 15%,'
    '#353535 15% 30%,#eee 30% 45%,#353535 45% 60%,'
    '#eee 60% 75%,#353535 75% 90%,#eee 90%);border-radius:4px;'
    'margin:8px 0 22px"></div>',
    unsafe_allow_html=True,
)

DEFAULT_WEIGHTS = "runs/detect/runs/detect/sponsorvision_v7-2/weights/best.pt"
BASE = Path(__file__).resolve().parent
SPONSOR_LOGOS = BASE / "assets" / "sponsor_logos"


def open_local_file_picker():
    """Open a native OS picker on the Windows PC running Streamlit.

    Tkinter dialogs are server-side: this is intentionally for a local desktop
    installation and cannot browse a remote viewer's PC.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        try:
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()
            selected = filedialog.askopenfilename(
                parent=root,
                title="Select SponsorVision match video",
                filetypes=[
                    ("Video files", "*.mp4 *.mov *.mkv *.avi"),
                    ("All files", "*.*"),
                ],
            )
            return selected, None
        finally:
            root.destroy()
    except Exception as exc:
        return "", (
            "The Windows file dialog could not open on the Streamlit computer. "
            f"You can still paste a video path or select Upload video. ({exc})"
        )


if "sv_local_video_path" not in st.session_state:
    st.session_state["sv_local_video_path"] = ""

# Start on an empty dashboard instead of reopening a previously saved run.
# Old runs stay on disk and can be selected from the run-history dropdown.
if "sv_pick_run" not in st.session_state:
    st.session_state["sv_pick_run"] = "__none__"

with st.sidebar:
    st.markdown("## SPONSORVISION")
    st.caption("MATCH ANALYTICS")
    st.divider()
    st.markdown("### Match setup")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0)
    source = st.radio("Video source", ["Local video path", "Upload video"])
    video_path = None
    video_upload = None
    if source == "Local video path":
        st.caption("Choose a file directly from this Windows computer (no upload needed).")
        if st.button("Browse Windows files...", key="sv_browse_local_video", use_container_width=True):
            selected, error = open_local_file_picker()
            if selected:
                # Keep the selected path across Streamlit reruns and tab changes.
                st.session_state["sv_local_video_path"] = selected
            elif error:
                st.warning(error)
        raw = st.text_input(
            "Selected video file path",
            key="sv_local_video_path",
            help="You can still paste a full path here. For remote Streamlit servers, use Upload video instead.",
        )
        if raw.strip():
            candidate = Path(raw.strip().strip('"').strip("'"))
            if not candidate.is_absolute():
                candidate = BASE / candidate
            if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
                video_path = candidate
                st.success(f"Ready: {candidate.name}")
            else:
                st.warning("Video not found. Click Browse Windows files or enter a valid path.")
    else:
        st.caption("Uses the browser file picker and temporarily uploads the video to this Streamlit server.")
        video_upload = st.file_uploader("Choose a video", type=["mp4", "mov", "mkv", "avi"])

    st.divider()
    st.markdown("### Detection settings")
    weights_path = st.text_input("Custom YOLO weights", value=DEFAULT_WEIGHTS)
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    interval = st.slider("Sample every (seconds)", 0.2, 3.0, 1.0, 0.2)
    with st.expander("Advanced · optional ORB matching"):
        logos = st.file_uploader(
            "Reference logo images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )
        min_inliers = st.slider("Minimum ORB inlier matches", 6, 30, 12)
        st.caption("Leave reference images empty to use your trained YOLO11n model only.")

    model_file = Path(weights_path.strip().strip('"').strip("'")) if weights_path.strip() else None
    if model_file is not None and not model_file.is_absolute():
        model_file = BASE / model_file
    if model_file and model_file.is_file():
        st.success("YOLO model found")
    elif weights_path.strip():
        st.error("Model file not found. Check the path.")
    st.divider()
    if st.button("Clear current analysis", key="sv_clear_analysis", use_container_width=True):
        # Hide current results without deleting any saved CSV reports.
        st.session_state["sv_pick_run"] = "__none__"
        st.rerun()


def sponsor_name_from_reference(filename):
    """Align common reference-image filenames with the seven model labels."""
    stem = Path(filename).stem.lower().replace("_", " ").replace("-", " ")
    compact = "".join(ch for ch in stem if ch.isalnum())
    if "raywhite" in compact or compact.startswith("rayw"):
        return "rayw"
    if "kgm" in compact or "ssangyong" in compact:
        return "kgm"
    if "primo" in compact:
        return "primo"
    if "latrobe" in compact or "latrobefinancial" in compact:
        return "la trobe"
    for name in ("kfc", "emirates", "nike"):
        if name in compact:
            return name
    return Path(filename).stem


def build_references(files):
    orb = cv2.ORB_create(nfeatures=3000)
    refs = []
    for f in files or []:
        arr = np.frombuffer(f.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            st.warning(f"Could not read {f.name}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            st.warning(f"{f.name}: insufficient ORB features; custom YOLO training may be necessary")
            continue
        refs.append((sponsor_name_from_reference(f.name), kp, des, image.shape[:2]))
    return refs


def orb_detect(frame, refs, threshold):
    if not refs:
        return []
    orb = cv2.ORB_create(nfeatures=4500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_des = orb.detectAndCompute(gray, None)
    if frame_des is None or len(frame_des) < 2:
        return []
    fh, fw = frame.shape[:2]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    detections = []
    for sponsor, ref_kp, ref_des, (rh, rw) in refs:
        pairs = matcher.knnMatch(ref_des, frame_des, k=2)
        good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.7 * b.distance]
        if len(good) < threshold:
            continue
        src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)
        if H is None or mask is None:
            continue
        inliers = int(mask.sum())
        if inliers < threshold or inliers / len(good) < 0.5:
            continue
        corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected.astype(np.float32)):
            continue
        x1, y1 = np.floor(projected.min(axis=0)).astype(int)
        x2, y2 = np.ceil(projected.max(axis=0)).astype(int)
        w, h = x2 - x1, y2 - y1
        area = w * h
        if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh or w < 12 or h < 12:
            continue
        if area > 0.20 * fw * fh or area < 0.00001 * fw * fh:
            continue
        # Avoid very stretched, degenerate homographies; these are frequent false positives.
        if max(w / h, h / w) > 12:
            continue
        detections.append(dict(sponsor=sponsor, x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                               method="ORB", score=float(inliers / len(good)), inlier_matches=inliers))
    return detections


def load_yolo(path):
    if not path.strip():
        return None
    p = Path(path.strip().strip('"').strip("'"))
    if not p.is_absolute():
        p = BASE / p
    if not p.is_file() or p.suffix.lower() != ".pt":
        raise ValueError("Custom YOLO weights file (.pt) was not found")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install optional YOLO dependency: python -m pip install ultralytics") from exc
    return YOLO(str(p))


def yolo_detect(model, frame, threshold):
    if model is None:
        return []
    result = model.predict(frame, conf=threshold, verbose=False)[0]
    found = []
    for box in result.boxes:
        cls = int(box.cls.item())
        name = str(result.names[cls])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        found.append(dict(sponsor=name, x1=x1, y1=y1, x2=x2, y2=y2,
                          method="YOLO", score=float(box.conf.item()), inlier_matches=None))
    return found


def iou(a, b):
    xx1, yy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    xx2, yy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    overlap = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    aa = max(0, a['x2']-a['x1']) * max(0, a['y2']-a['y1'])
    bb = max(0, b['x2']-b['x1']) * max(0, b['y2']-b['y1'])
    return overlap / (aa + bb - overlap) if aa + bb - overlap > 0 else 0


def merge_detections(items):
    # Keep the highest-scoring box when two methods find the same sponsor and region.
    chosen = []
    for item in sorted(items, key=lambda d: d['score'], reverse=True):
        if not any(item['sponsor'].casefold() == prev['sponsor'].casefold() and iou(item, prev) > 0.45 for prev in chosen):
            chosen.append(item)
    return chosen


def process_video(path, refs, model, q, step, inliers, conf, video_name):
    """Run existing YOLO + optional ORB detection once and record every sampled time."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video. Try an H.264 MP4 file.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (math.isfinite(fps) and fps > 0 and math.isfinite(frame_count) and frame_count > 0):
            raise ValueError("Video FPS or frame count unavailable")
        duration = frame_count / fps
        attempted = math.ceil(duration / step)
        model_names = []
        if model is not None:
            model_names = [str(model.names[k]) for k in sorted(model.names)]
        sponsors = list(dict.fromkeys(model_names + [ref[0] for ref in refs]))
        if not sponsors:
            raise ValueError("No model or usable reference images provided")

        rows = []
        presence_rows = []
        decoded = 0
        progress = st.progress(0)
        message = st.empty()
        preview = st.empty()
        preview_count = 0

        for i in range(attempted):
            timestamp = i * step
            remaining = max(0.0, min(step, duration - timestamp))
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            current = []
            if ok:
                decoded += 1
                current = merge_detections(
                    orb_detect(frame, refs, inliers) + yolo_detect(model, frame, conf)
                )
                for d in current:
                    rows.append(dict(
                        video=video_name, quarter=q, sample_index=i,
                        timestamp_sec=round(timestamp, 3), **d
                    ))
                if current and preview_count < 8:
                    annotated = frame.copy()
                    for d in current:
                        cv2.rectangle(annotated, (d["x1"], d["y1"]),
                                      (d["x2"], d["y2"]), (0, 255, 0), 3)
                        cv2.putText(annotated, f"{d['sponsor']} {d['method']}",
                                    (d["x1"], max(25, d["y1"] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    height, width = annotated.shape[:2]
                    if width > 960:
                        annotated = cv2.resize(annotated, (960, int(height * 960 / width)))
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                  caption=f"Example detection at {timestamp:.1f}s")
                    preview_count += 1

            present = {d["sponsor"] for d in current}
            presence_rows.append({
                "video": video_name,
                "quarter": q,
                "sample_index": i,
                "timestamp_sec": round(timestamp, 3),
                "sample_duration_sec": round(remaining, 6) if ok else 0.0,
                "decoded": int(bool(ok)),
                **{sponsor: int(sponsor in present) for sponsor in sponsors},
            })

            if i % 10 == 0 or i == attempted - 1:
                progress.progress((i + 1) / attempted)
                message.caption(
                    f"Sample positions: {i+1:,}/{attempted:,} | decoded: {decoded:,} "
                    f"| detections: {len(rows):,}"
                )

        df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
        presence = pd.DataFrame(presence_rows, columns=[
            "video", "quarter", "sample_index", "timestamp_sec",
            "sample_duration_sec", "decoded", *sponsors
        ])
        return df, presence, sponsors, decoded, attempted, duration
    finally:
        cap.release()


ready = (video_path is not None or video_upload is not None) and (
    bool(logos) or bool(model_file is not None and model_file.is_file())
)

st.markdown("### Run sponsor analysis")
st.caption("Select a video and detection settings in the sidebar. Each completed run is saved automatically.")
if st.button("▶ ANALYSE VIDEO NOW", key="main_analyse", use_container_width=True,
             type="primary", disabled=not ready):
    try:
        refs = build_references(logos)
        model = load_yolo(weights_path)
        if not refs and model is None:
            st.error("Supply trained YOLO weights or usable ORB reference images.")
        else:
            if video_path is not None:
                video_name = video_path.name
                df, presence, sponsors, decoded, attempted, duration = process_video(
                    video_path, refs, model, quarter, interval, min_inliers,
                    conf_threshold, video_name
                )
            else:
                video_name = Path(video_upload.name).name
                with tempfile.TemporaryDirectory() as folder:
                    temporary_video = Path(folder) / ("input" + Path(video_name).suffix.lower())
                    video_upload.seek(0)
                    with temporary_video.open("wb") as output:
                        while True:
                            chunk = video_upload.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                    df, presence, sponsors, decoded, attempted, duration = process_video(
                        temporary_video, refs, model, quarter, interval, min_inliers,
                        conf_threshold, video_name
                    )

            intervals_df, summary = generate_reports(
                df, presence, sponsors, video_name, quarter, duration
            )
            folder = save_analysis(
                df, presence, intervals_df, summary, video_name,
                str(model_file) if model is not None else "ORB only",
                quarter, interval, conf_threshold, duration, decoded,
                attempted, [f.name for f in logos or []],
            )
            st.session_state["sv_pick_run"] = folder.name
            st.success(
                f"Completed: {decoded:,}/{attempted:,} samples, {len(df):,} logo boxes. "
                "Four CSV reports and a ZIP were saved automatically."
            )
            st.caption(f"Saved to: {folder}")
            if decoded < attempted:
                st.warning("Some sampled positions could not be decoded and are excluded from exposure.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

runs = list_analyses()
run_by_name = {run.name: run for run in runs}

if runs:
    current = st.session_state.get("sv_pick_run")

    if current not in run_by_name and current != "__none__":
        st.session_state["sv_pick_run"] = "__none__"

    def label_for_run(run_id):
        if run_id == "__none__":
            return "No analysis selected"

        folder = run_by_name[run_id]
        import json
        metadata = json.loads(
            (folder / "metadata.json").read_text(encoding="utf-8")
        )
        return (
            f"{metadata['video']} · "
            f"Q{metadata['quarter']} · "
            f"{metadata['timestamp_local']}"
        )

    st.selectbox(
        "View saved video analysis",
        options=["__none__"] + list(run_by_name),
        format_func=label_for_run,
        key="sv_pick_run",
    )

    if st.session_state["sv_pick_run"] == "__none__":
        analysis = None
    else:
        try:
            analysis = load_analysis(
                run_by_name[st.session_state["sv_pick_run"]]
            )
        except Exception as exc:
            analysis = None
            st.error(f"Cannot load saved analysis: {exc}")
else:
    analysis = None

st.divider()
overview_tab, analysis_tab, insights_tab, data_tab = st.tabs(
    ["Overview", "Video analysis", "Sponsor insights", "Data & exports"]
)

with overview_tab:
    st.subheader("Match intelligence overview")

    # One source of truth: the currently selected (or newly completed) saved run.
    # Do not use old sv_df/sv_step variables, which are no longer populated.
    if analysis is None:
        st.info("No video analysis selected. Choose a match video in the sidebar and click **ANALYSE VIDEO NOW**.")
    else:
        df = analysis["detections"]
        summary = analysis["summary"]
        metadata = analysis["metadata"]

        sponsor_count = int(df["sponsor"].nunique()) if not df.empty else 0
        visible_seconds = float(summary["sampled_visible_seconds"].sum()) if not summary.empty else 0.0

        a, b, c, d = st.columns(4)
        a.metric("Sponsors detected", sponsor_count)
        b.metric("Detection boxes", f"{len(df):,}")
        c.metric("Sampled visibility total", f"{visible_seconds:,.1f}s")
        d.metric("Quarter analysed", f"Q{metadata['quarter']}")

        if sponsor_count > 0 and not summary.empty:
            st.subheader("Sponsor visibility ranking")
            ranking = summary.set_index("sponsor")["sampled_visible_seconds"]
            st.bar_chart(ranking, horizontal=True)
        else:
            st.info("Analysis completed, but no sponsor logos were detected at the chosen threshold.")

with analysis_tab:
    st.subheader("Analyse match footage")
    st.write("Choose a local video or upload footage using the sidebar, then click **ANALYSE VIDEO NOW** above.")
    st.caption("The application saves four CSVs and one ZIP per analysed video. Videos are not permanently copied.")
    if analysis is not None:
        m = analysis["metadata"]
        st.write(f"**Selected video:** {m['video']} | Quarter {m['quarter']} | "
                 f"{m['decoded_samples']:,} decoded samples / {m['attempted_samples']:,} attempts")
        st.caption(f"Model: {m['model_path']} | Sample every {m['sample_every_seconds']:g}s "
                   f"| Confidence threshold: {m['confidence_threshold']:.2f}")
        if m["decoded_samples"] < m["attempted_samples"]:
            st.warning("Not all sample positions were decoded; exposure is based on successfully decoded samples.")

with insights_tab:
    st.subheader("Sponsor insights")
    if analysis is None:
        st.info("Run an analysis first to view sponsor insights.")
    else:
        df, summary = analysis["detections"], analysis["summary"]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Sampled visibility by sponsor")
            st.bar_chart(summary.set_index("sponsor")["sampled_visible_seconds"])
        with right:
            st.markdown("#### Detection boxes by sponsor")
            st.bar_chart(summary.set_index("sponsor")["total_boxes"])
        if not df.empty:
            st.markdown("#### Detection activity over time")
            timeline = df.copy()
            timeline["minute"] = (timeline["timestamp_sec"] // 60).astype(int)
            counts = timeline.groupby(["minute", "sponsor"]).size().unstack(fill_value=0)
            st.line_chart(counts)
        else:
            st.info("No sponsor detections found in this video at the selected settings.")
        st.caption("Timeline counts bounding boxes, not continuous seconds of exposure.")
        st.markdown("#### Exposure estimates")
        st.dataframe(summary, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Detection data & exports")
    if analysis is None:
        st.info("Analyse a video first. Results are saved automatically and remain available after restarting Streamlit.")
    else:
        df, summary = analysis["detections"], analysis["summary"]
        folder = analysis["folder"]
        st.success(f"CSV reports saved for: {analysis['metadata']['video']}")
        st.caption(f"Report folder: {folder}")

        with (folder / "all_csv_reports.zip").open("rb") as file:
            st.download_button(
                "Download all 4 CSV reports (ZIP)", file, file_name=f"{folder.name}_reports.zip",
                mime="application/zip", use_container_width=True, type="primary",
            )
        st.markdown("#### Individual CSV downloads")
        report_files = [
            ("detections.csv", "All bounding-box detections"),
            ("frame_presence.csv", "One row for every sampled frame"),
            ("exposure_intervals.csv", "Estimated start and end of appearance runs"),
            ("exposure_summary.csv", "Total estimated visibility per sponsor"),
        ]
        for filename, label in report_files:
            with (folder / filename).open("rb") as file:
                st.download_button(
                    f"Download {filename} — {label}",
                    file, file_name=f"{folder.name}_{filename}", mime="text/csv",
                    key=f"download_{folder.name}_{filename}", use_container_width=True,
                )

        st.markdown("#### Detection table")
        if not df.empty:
            # Include all model classes, even if a sponsor was not detected in this video.
            model_classes = [str(name) for name in summary["sponsor"].dropna().tolist()]
            observed_classes = [str(name) for name in df["sponsor"].dropna().unique()]
            options = list(dict.fromkeys(model_classes + observed_classes))
            counts_by_sponsor = df["sponsor"].value_counts().to_dict()
            labels = {
                "kfc": "KFC", "emirates": "Emirates", "nike": "Nike",
                "la trobe": "La Trobe", "primo": "Primo",
                "rayw": "Ray White", "kgm": "KGM SsangYong",
            }

            st.markdown("**Filter sponsors by logo**")
            st.caption("Tick sponsors to show their detections. Numbers are detected boxes, not exposure time.")

            # Selection is unique to each saved video run, so switching videos
            # doesn't unexpectedly reuse a different run's filter selection.
            def filter_key(sponsor):
                return f"sv_sponsor_filter_{folder.name}_{sponsor}"

            for sponsor in options:
                key = filter_key(sponsor)
                if key not in st.session_state:
                    st.session_state[key] = True

            select_col, clear_col, spacer = st.columns([1, 1, 3])
            with select_col:
                if st.button("Select all", key=f"sv_filter_all_{folder.name}", use_container_width=True):
                    for sponsor in options:
                        st.session_state[filter_key(sponsor)] = True
            with clear_col:
                if st.button("Clear all", key=f"sv_filter_clear_{folder.name}", use_container_width=True):
                    for sponsor in options:
                        st.session_state[filter_key(sponsor)] = False

            for start in range(0, len(options), 4):
                columns = st.columns(4)
                for col, sponsor in zip(columns, options[start:start + 4]):
                    with col:
                        with st.container(border=True):
                            logo_file = SPONSOR_LOGOS / f"{sponsor.replace(' ', '_')}.png"
                            if logo_file.is_file():
                                st.image(str(logo_file), use_container_width=True)
                            else:
                                st.caption("Logo image unavailable")
                            st.checkbox(
                                f"{labels.get(sponsor, sponsor.title())} ({int(counts_by_sponsor.get(sponsor, 0))})",
                                key=filter_key(sponsor),
                            )

            selected = [name for name in options if st.session_state[filter_key(name)]]
            filtered = df[df["sponsor"].isin(selected)]
            st.caption(f"Showing {len(filtered):,} of {len(df):,} detected boxes from {len(selected)} selected sponsors.")
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered detections CSV", filtered.to_csv(index=False).encode("utf-8"),
                f"{folder.name}_filtered_detections.csv", "text/csv",
            )
        else:
            st.info("No detected logo boxes, but presence and summary CSVs were still saved.")

        st.markdown("#### Exposure summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("#### Exposure intervals")
        st.dataframe(analysis["intervals"], use_container_width=True, hide_index=True)


