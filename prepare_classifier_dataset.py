from pathlib import Path
from PIL import Image
import shutil

DATASET_DIR = Path("dataset/NeuroVision_YOLO_v2")
OUTPUT_DIR = Path("classifier_dataset")

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No_Tumor",
    3: "Pituitary"
}

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    images = list(image_dir.glob("*.jpg"))

    print(f"\nProcessing {split}: {len(images)} images")

    copied = 0
    skipped = 0

    for image_path in images:

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            skipped += 1
            continue

        lines = label_path.read_text().strip().splitlines()

        if not lines:
            skipped += 1
            continue

        class_ids = set()

        for line in lines:
            parts = line.split()

            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                if class_id in CLASS_NAMES:
                    class_ids.add(class_id)

        if not class_ids:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")

        for class_id in class_ids:

            class_name = CLASS_NAMES[class_id]

            output_dir = OUTPUT_DIR / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / image_path.name

            image.save(output_path, quality=95)

        copied += 1

    print(f"Images copied: {copied}")
    print(f"Images skipped: {skipped}")

print("\nCLASSIFIER DATASET PREPARATION COMPLETE!")