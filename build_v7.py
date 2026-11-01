"""Build sponsorvision_v7 from three YOLO datasets.

Sources:
  v5      datasets/sponsorvision_v5          train + val
  rnd13   datasets/sponsorvision_v6          only the rnd13_ files (v5 files are skipped)
  cfc     collingwood/data/datasets          train + val + test

Labels are remapped by class name, so each source's own class IDs don't matter.
Exact duplicate images (same file content) are kept only once, from the first
source in the order above. Any test image that duplicates a train/val image is
dropped so the test split stays unseen.
"""
from pathlib import Path
from collections import Counter
import hashlib
import shutil

import cv2
import yaml

DEST = Path(r"C:\Users\saran\Desktop\collingwood_sponsorvision\datasets\sponsorvision_v7")
UNIFIED = ["kfc", "emirates", "nike", "la trobe", "primo", "rayw", "kgm"]
NAME_ALIASES = {"latrobe": "la trobe"}

SOURCES = [
    # (tag, folder, splits to use, file-name filter)
    ("v5", Path(r"C:\Users\saran\Desktop\collingwood_sponsorvision\datasets\sponsorvision_v5"),
     ["train", "val"], lambda name: True),
    ("rnd13", Path(r"C:\Users\saran\Desktop\collingwood_sponsorvision\datasets\sponsorvision_v6"),
     ["train", "val"], lambda name: name.startswith("rnd13_")),
    ("cfc", Path(r"C:\Users\saran\Desktop\collingwood\data\datasets"),
     ["train", "val", "test"], lambda name: True),
]

if DEST.exists():
    raise SystemExit("sponsorvision_v7 already exists. Nothing overwritten.")

unified_ids = {name: i for i, name in enumerate(UNIFIED)}


def source_names(folder):
    config = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def remap_label(text, id_map, source, label_path):
    out = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{label_path}:{line_no} expected 5 fields, got {len(parts)}")
        cls = int(parts[0])
        if cls not in id_map:
            raise SystemExit(f"{label_path}:{line_no} class id {cls} not in {source} names")
        coords = [float(v) for v in parts[1:]]
        if not all(0 <= v <= 1 for v in coords):
            raise SystemExit(f"{label_path}:{line_no} coordinates outside 0..1: {coords}")
        out.append(" ".join([str(id_map[cls])] + parts[1:]))
    return out


# Collect every candidate image first, so nothing is written if a check fails.
candidates = []  # (tag, split, image_path, label_lines)
for tag, folder, splits, keep in SOURCES:
    if not folder.exists():
        raise SystemExit(f"Missing source folder: {folder}")
    names = source_names(folder)
    id_map = {}
    for i, name in enumerate(names):
        key = NAME_ALIASES.get(name.lower(), name.lower())
        if key not in unified_ids:
            raise SystemExit(f"{tag}: class '{name}' is not in the unified list")
        id_map[i] = unified_ids[key]
    print(f"{tag} names -> unified:", {names[i]: id_map[i] for i in id_map})

    for split in splits:
        # Accept both layouts: images/<split> + labels/<split>, or <split>/images + <split>/labels.
        if (folder / "images" / split).exists():
            image_dir, label_dir = folder / "images" / split, folder / "labels" / split
        else:
            image_dir, label_dir = folder / split / "images", folder / split / "labels"
        if not image_dir.exists():
            raise SystemExit(f"Missing folder: {image_dir}")
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not keep(image_path.name):
                continue
            label_path = label_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                raise SystemExit(f"Missing label for {image_path}")
            if cv2.imread(str(image_path)) is None:
                raise SystemExit(f"Cannot read image: {image_path}")
            lines = remap_label(label_path.read_text(encoding="utf-8"), id_map, tag, label_path)
            candidates.append((tag, split, image_path, lines))

print("Candidate images:", len(candidates))

# Drop exact duplicates. Earlier sources win; test images that repeat a train/val image are dropped.
seen = {}  # hash -> split it was kept in
kept, dropped = [], Counter()
for tag, split, image_path, lines in candidates:
    digest = md5(image_path)
    if digest in seen:
        dropped[f"{tag}/{split} duplicate of {seen[digest]}"] += 1
        continue
    seen[digest] = f"{tag}/{split}"
    kept.append((tag, split, image_path, lines, digest))

# Test images must not appear in train or val.
trainval_hashes = {d for tag, split, _, _, d in kept if split in {"train", "val"}}
final = []
for tag, split, image_path, lines, digest in kept:
    if split == "test" and digest in trainval_hashes:
        dropped["cfc/test repeats a train/val image"] += 1
        continue
    final.append((tag, split, image_path, lines))

print("Duplicates dropped:", dict(dropped) or "none")
print("Final split sizes:", dict(Counter(split for _, split, _, _ in final)))
print("Final per-class boxes:", dict(sorted(Counter(
    UNIFIED[int(line.split()[0])] for _, _, _, lines in final for line in lines
).items())))

# Write the dataset.
for split in ("train", "val", "test"):
    (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

for tag, split, image_path, lines in final:
    stem = f"{tag}_{image_path.stem}"
    target_image = DEST / "images" / split / f"{stem}{image_path.suffix.lower()}"
    target_label = DEST / "labels" / split / f"{stem}.txt"
    if target_image.exists() or target_label.exists():
        raise SystemExit(f"Duplicate output name: {stem}")
    shutil.copy2(image_path, target_image)
    target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

config = {
    "path": str(DEST.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {i: n for i, n in enumerate(UNIFIED)},
}
(DEST / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

print("sponsorvision_v7 created:", DEST)
