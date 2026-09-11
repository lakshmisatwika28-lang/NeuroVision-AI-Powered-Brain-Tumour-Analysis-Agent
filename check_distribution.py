from pathlib import Path
from collections import Counter

ROOT = Path("dataset/NeuroVision_YOLO_v2")

CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

print("=" * 60)
print("NEUROVISION DATASET DISTRIBUTION CHECK")
print("=" * 60)

for split in ["train", "val", "test"]:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    image_count = len(list(image_dir.glob("*.jpg")))
    counts = Counter()

    for label_file in label_dir.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))

                if class_id in CLASSES:
                    counts[class_id] += 1

    print(f"\n{split.upper()}")
    print("-" * 40)

    print(f"Images : {image_count}")
    print(f"Boxes  : {sum(counts.values())}")

    for class_id, name in CLASSES.items():
        print(f"{class_id} - {name:12s}: {counts[class_id]}")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)