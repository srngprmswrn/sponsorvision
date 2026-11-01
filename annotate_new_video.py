
from pathlib import Path
import cv2
import json

# -------------------------------------------
# SETTINGS
# -------------------------------------------

IMAGE_DIR = Path("training_sources/new_video/data")
OUTPUT_DIR = Path("training_sources/new_video")
OUTPUT_FILE = OUTPUT_DIR / "sponsor_annotations.json"

# Add sponsor names here whenever needed.
SPONSORS = [
    "kfc",
    "emirates",
    "nike",
    "ray white",
    "primo",
    "kgm",
    "la trobe",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    raise SystemExit(f"No images found in {IMAGE_DIR}")

# -------------------------------------------
# LOAD PREVIOUS ANNOTATIONS
# -------------------------------------------

if OUTPUT_FILE.exists():
    annotations = json.loads(
        OUTPUT_FILE.read_text(encoding="utf-8")
    )
else:
    annotations = {}

print(f"Images found: {len(images)}")
print(f"Already reviewed: {len(annotations)}")

print("\nCONTROLS")
print("Draw rectangle with mouse, then ENTER/SPACE.")
print("After drawing, enter the sponsor number in terminal.")
print("Press C or ESC in the rectangle window to finish an image.")
print("Enter 'u' in terminal to undo the last box.")
print("Enter 's' in terminal to stop and save.")
print("Enter 'n' in terminal to finish image.")
print("Enter 'r' in terminal to reset current image.")
print("\nInclude jersey logos, advertising boards,")
print("goalpost padding, on-field signage and graphics.")
print("Only label sponsors that are actually visible.\n")

WINDOW = "SponsorVision Annotation"


def save_annotations():
    OUTPUT_FILE.write_text(
        json.dumps(annotations, indent=2),
        encoding="utf-8"
    )


def show_preview(image, boxes, scale, filename):
    display = cv2.resize(
        image,
        (
            round(image.shape[1] * scale),
            round(image.shape[0] * scale),
        )
    )

    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        sponsor = item["sponsor"]

        p1 = (round(x1 * scale), round(y1 * scale))
        p2 = (round(x2 * scale), round(y2 * scale))

        cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            display,
            sponsor,
            (p1[0], max(18, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        display,
        f"{filename} | boxes: {len(boxes)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    return display


# -------------------------------------------
# ANNOTATION LOOP
# -------------------------------------------

stop_requested = False

for index, image_path in enumerate(images, start=1):

    if image_path.name in annotations:
        print(f"Skipping completed: {image_path.name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot read: {image_path.name}")
        continue

    height, width = image.shape[:2]
    scale = min(1.0, 1200 / width, 750 / height)

    boxes = []

    print(
        f"\nImage {index}/{len(images)}: "
        f"{image_path.name}"
    )

    while True:
        preview = show_preview(
            image, boxes, scale, image_path.name
        )

        roi = cv2.selectROI(
            WINDOW,
            preview,
            showCrosshair=True,
            fromCenter=False,
        )

        x, y, w, h = roi

        # No rectangle: finish the image.
        if w <= 0 or h <= 0:
            break

        x1 = max(0, round(x / scale))
        y1 = max(0, round(y / scale))
        x2 = min(width, round((x + w) / scale))
        y2 = min(height, round((y + h) / scale))

        print("\nChoose sponsor:")

        for number, sponsor in enumerate(SPONSORS, start=1):
            print(f"{number}. {sponsor}")

        print("u. Undo last box")
        print("n. Finish image")
        print("r. Reset current image")
        print("s. Stop and save")

        choice = input("Choice: ").strip().lower()

        if choice == "u":
            if boxes:
                removed = boxes.pop()
                print("Removed:", removed)
            continue

        if choice == "r":
            boxes = []
            print("Current image cleared.")
            continue

        if choice == "n":
            break

        if choice == "s":
            stop_requested = True
            break

        if not choice.isdigit():
            print("Invalid choice. Box not added.")
            continue

        sponsor_index = int(choice) - 1

        if not 0 <= sponsor_index < len(SPONSORS):
            print("Invalid sponsor number. Box not added.")
            continue

        sponsor = SPONSORS[sponsor_index]

        boxes.append({
            "sponsor": sponsor,
            "box": [x1, y1, x2, y2],
        })

        print(f"Added: {sponsor} {[x1, y1, x2, y2]}")

    # If stopped, preserve any boxes from this image.
    # Do not mark a partially reviewed image complete.
    if stop_requested:
        if boxes:
            print(
                "Stopping. Current unfinished image "
                "was not saved; previous images are safe."
            )
        break

    annotations[image_path.name] = boxes
    save_annotations()

    print(
        f"Saved {image_path.name}: "
        f"{len(boxes)} logos"
    )

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
save_annotations()

total_boxes = sum(
    len(items) for items in annotations.values()
)

print("\nFinished!")
print(f"Reviewed images: {len(annotations)}")
print(f"Total sponsor boxes: {total_boxes}")
print(f"Saved to: {OUTPUT_FILE}")
