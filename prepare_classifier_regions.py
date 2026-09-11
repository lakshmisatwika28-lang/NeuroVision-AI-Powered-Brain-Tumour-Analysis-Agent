from pathlib import Path
from PIL import Image
import shutil

ROOT = Path("dataset/NeuroVision_YOLO_v2")
OUTPUT = Path("classifier_regions")

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No_Tumor",
    3: "Pituitary"
}

SPLITS = ["train", "val", "test"]

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir(parents=True)

total_crops = 0

for split in SPLITS:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    print(f"\nProcessing {split}...")

    split_count = 0
    skipped = 0

    for image_path in image_dir.glob("*.jpg"):

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            skipped += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size

            with open(label_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            crop_index = 0

            for line in lines:

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                # No Tumor has no bounding box
                if class_id == 2:
                    continue

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # YOLO normalized coordinates -> pixel coordinates
                x1 = int((x_center - box_width / 2) * img_width)
                y1 = int((y_center - box_height / 2) * img_height)
                x2 = int((x_center + box_width / 2) * img_width)
                y2 = int((y_center + box_height / 2) * img_height)

                # Keep coordinates inside image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image.crop((x1, y1, x2, y2))

                class_name = CLASS_NAMES.get(class_id)

                if class_name is None:
                    continue

                class_dir = OUTPUT / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                output_name = f"{image_path.stem}_crop{crop_index}.jpg"
                output_path = class_dir / output_name

                crop.save(output_path, quality=95)

                split_count += 1
                total_crops += 1
                crop_index += 1

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            skipped += 1

    print(f"Crops created: {split_count}")
    print(f"Images skipped: {skipped}")

print("\nCLASSIFIER REGION DATASET COMPLETE!")
print(f"Total tumor crops created: {total_crops}")