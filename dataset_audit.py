from pathlib import Path
from collections import Counter

DATASET = Path("dataset/NeuroVision_YOLO")

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

for split in ["train", "val"]:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = list(image_dir.glob("*.jpg"))

    class_counts = Counter()
    total_boxes = 0
    missing_labels = []
    empty_labels = []
    bad_labels = []

    for image_path in images:

        label_path = label_dir / f"{image_path.stem}.txt"

        # Check whether label exists
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        with open(label_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Empty label file
        if len(lines) == 0:
            empty_labels.append(image_path.name)
            continue

        for line in lines:

            parts = line.split()

            # YOLO format must contain 5 values
            if len(parts) != 5:
                bad_labels.append((image_path.name, line))
                continue

            try:
                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:]]

                # Check class ID
                if class_id not in CLASS_NAMES:
                    bad_labels.append((image_path.name, line))
                    continue

                # Coordinates should be between 0 and 1
                if not all(0 <= x <= 1 for x in coordinates):
                    bad_labels.append((image_path.name, line))
                    continue

                class_counts[class_id] += 1
                total_boxes += 1

            except ValueError:
                bad_labels.append((image_path.name, line))

    print("\n" + "=" * 50)
    print(f"{split.upper()} DATASET AUDIT")
    print("=" * 50)

    print(f"Images              : {len(images)}")
    print(f"Total bounding boxes: {total_boxes}")

    print("\nBounding boxes by class:")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id} - {class_name:<12}: {class_counts[class_id]}")

    print("\nProblems found:")
    print(f"  Missing labels : {len(missing_labels)}")
    print(f"  Empty labels   : {len(empty_labels)}")
    print(f"  Bad labels     : {len(bad_labels)}")

    if missing_labels:
        print("\nMissing label files:")
        for name in missing_labels[:10]:
            print(" ", name)

    if empty_labels:
        print("\nEmpty label files:")
        for name in empty_labels[:10]:
            print(" ", name)

    if bad_labels:
        print("\nBad labels:")
        for name, line in bad_labels[:10]:
            print(" ", name, "->", line)

print("\n" + "=" * 50)
print("DATASET AUDIT COMPLETE")
print("=" * 50)