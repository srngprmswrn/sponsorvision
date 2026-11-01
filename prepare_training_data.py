
from pathlib import Path
from collections import Counter
from PIL import Image
import json
import shutil

ROOT = Path("training_sources")
OUTPUT = Path("sponsor_dataset")

# Keep entire video clips in separate splits.
DATASETS = {
    "rnd15": "train",
    "rnd17": "val",
}

CLASSES = [
    "kfc",
    "nike",
    "latrobe",
    "rayw",
    "kgm",
    "primo",
]

CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}

# Only convert labels that were explicitly annotated.
LABEL_ALIASES = {
    "kfc": "kfc",
    "nike": "nike",
    "latrobe": "latrobe",
    "la trobe": "latrobe",
    "rayw": "rayw",
    "ray white": "rayw",
    "kgm": "kgm",
    "primo": "primo",
}

if OUTPUT.exists():
    raise SystemExit(
        f"{OUTPUT} already exists. Rename or move it "
        "before running this script again."
    )

counts = Counter()

for dataset, split in DATASETS.items():
    folder = ROOT / dataset
    annotation_file = folder / "annotations.json"

    if not annotation_file.exists():
        raise FileNotFoundError(annotation_file)

    data = json.loads(annotation_file.read_text(encoding="utf-8"))

    shapes_by_frame = {}

    for item in data:
        for shape in item.get("shapes", []):
            if shape.get("type") != "rectangle":
                continue

            frame = int(shape["frame"])
            shapes_by_frame.setdefault(frame, []).append(shape)

    images_out = OUTPUT / "images" / split
    labels_out = OUTPUT / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Use only frames with annotations for this first dataset.
    for frame, shapes in sorted(shapes_by_frame.items()):
        image_path = folder / "data" / f"frame_{frame + 1:05d}.jpg"

        if not image_path.exists():
            print("Missing image:", image_path)
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        lines = []

        for shape in shapes:
            original_label = str(shape["label"]).strip().lower()
            label = LABEL_ALIASES.get(original_label)

            if label is None:
                print("Unknown sponsor label:", original_label)
                continue

            x1, y1, x2, y2 = map(float, shape["points"])

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            # Clip boxes to image boundaries.
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))

            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid box:", dataset, frame)
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            lines.append(
                f"{CLASS_IDS[label]} "
                f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

            counts[label] += 1

        if not lines:
            continue

        stem = f"{dataset}_frame_{frame + 1:05d}"

        shutil.copy2(
            image_path,
            images_out / f"{stem}.jpg"
        )

        (labels_out / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

    print(f"Finished {dataset} -> {split}")

yaml_text = f"""path: {OUTPUT.resolve().as_posix()}
train: images/train
val: images/val

names:
"""

for i, name in enumerate(CLASSES):
    yaml_text += f"  {i}: {name}\n"

(OUTPUT / "data.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)

print("\nSponsor annotations:")
for label, count in sorted(counts.items()):
    print(f"  {label}: {count}")

print("\nDataset ready:", OUTPUT.resolve())
