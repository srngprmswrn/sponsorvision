
import cv2
import json
from pathlib import Path

IMAGE_DIR = Path("training_sources/rnd15/data")
OUTPUT = Path("collingwood_v4_annotations.json")

FRAMES = [
    10, 11, 20, 37, 41, 43,
    44, 81, 104, 110, 120
]

LABELS = {
    ord("k"): "kfc",
    ord("n"): "nike"
}

if OUTPUT.exists():
    annotations = json.loads(OUTPUT.read_text())
else:
    annotations = {}

for frame in FRAMES:
    filename = f"frame_{frame:05d}.jpg"
    image = cv2.imread(str(IMAGE_DIR / filename))

    if image is None:
        print("Missing:", filename)
        continue

    annotations.setdefault(filename, [])
    current_label = "kfc"
    start = None

    def mouse(event, x, y, flags, param):
        global start

        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and start:
            x1, y1 = start
            x2, y2 = x, y

            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                annotations[filename].append({
                    "label": current_label,
                    "box": [
                        min(x1, x2), min(y1, y2),
                        max(x1, x2), max(y1, y2)
                    ]
                })
            start = None

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate", mouse)

    while True:
        display = image.copy()

        for item in annotations[filename]:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display, item["label"], (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.putText(
            display,
            f"{filename} | {current_label} | K=KFC N=Nike U=Undo S=Save/Next Q=Quit",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        cv2.imshow("Annotate", display)
        key = cv2.waitKey(30) & 0xFF

        if key in LABELS:
            current_label = LABELS[key]
        elif key == ord("u") and annotations[filename]:
            annotations[filename].pop()
        elif key in (ord("s"), ord("q")):
            OUTPUT.write_text(json.dumps(annotations, indent=2))
            if key == ord("q"):
                cv2.destroyAllWindows()
                print("Progress saved. You can resume later.")
                raise SystemExit
            break

    cv2.destroyAllWindows()
    print("Saved:", filename)

print("Finished! Annotations saved to:", OUTPUT)
