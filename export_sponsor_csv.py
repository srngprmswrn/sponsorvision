
from pathlib import Path
import argparse
import csv
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

MODEL = (
    BASE / "runs" / "detect" / "runs" / "detect"
    / "sponsorvision_v7-2" / "weights" / "best.pt"
)

OUTPUT = BASE / "output" / "sponsor_csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if not MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    if not Path(args.video).is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    if args.fps <= 0:
        raise ValueError("FPS must be positive")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL))
    names = [model.names[i] for i in sorted(model.names)]

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        cap.release()
        raise RuntimeError("Invalid source FPS")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = (
        frame_count / source_fps if frame_count > 0 else None
    )

    stride = max(1, round(source_fps / args.fps))
    interval = stride / source_fps

    counts = {name: 0 for name in names}
    seconds = {name: 0.0 for name in names}

    frame_id = 0
    processed = 0

    with (
        open(OUTPUT / "detections.csv", "w", newline="") as f1,
        open(OUTPUT / "frame_presence.csv", "w", newline="") as f2
    ):
        detections = csv.writer(f1)
        presence = csv.writer(f2)

        detections.writerow([
            "frame_id", "timestamp_sec", "logo",
            "confidence", "x1", "y1", "x2", "y2"
        ])

        presence.writerow(
            ["frame_id", "timestamp_sec"] + names
        )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_id % stride != 0:
                    frame_id += 1
                    continue

                timestamp = frame_id / source_fps

                result = model.predict(
                    frame,
                    imgsz=640,
                    conf=args.conf,
                    device=0,
                    verbose=False
                )[0]

                found = set()

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    name = model.names[class_id]
                    confidence = float(box.conf[0])

                    coords = [
                        round(float(v), 2)
                        for v in box.xyxy[0].tolist()
                    ]

                    found.add(name)

                    detections.writerow([
                        frame_id,
                        round(timestamp, 3),
                        name,
                        round(confidence, 4),
                        *coords
                    ])

                presence.writerow([
                    frame_id,
                    round(timestamp, 3),
                    *[int(name in found) for name in names]
                ])

                sample_duration = interval
                if video_duration is not None:
                    sample_duration = max(
                        0.0,
                        min(interval, video_duration - timestamp)
                    )

                for name in found:
                    counts[name] += 1
                    seconds[name] += sample_duration

                processed += 1

                if processed % 500 == 0:
                    print(f"Analysed {processed} sampled frames")

                frame_id += 1

        finally:
            cap.release()

    with open(
        OUTPUT / "exposure_summary.csv", "w", newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "logo", "frames_visible",
            "estimated_seconds", "estimated_minutes",
            "percent_sampled_frames"
        ])

        for name in names:
            writer.writerow([
                name,
                counts[name],
                round(seconds[name], 3),
                round(seconds[name] / 60, 3),
                round(
                    100 * counts[name] / processed, 2
                ) if processed else 0
            ])

    print("\nCSV EXPORT COMPLETE")
    print(f"Sampled frames: {processed}")
    print(f"Output folder: {OUTPUT}")

    for name in names:
        print(f"{name}: {seconds[name]:.2f} seconds")


if __name__ == "__main__":
    main()
