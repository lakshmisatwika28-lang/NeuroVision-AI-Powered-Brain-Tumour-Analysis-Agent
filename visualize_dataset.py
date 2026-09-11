from pathlib import Path
import random
from PIL import Image, ImageDraw

# Dataset location
DATASET = Path("dataset/NeuroVision_YOLO")

IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

OUTPUT_DIR = Path("dataset_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Pick 12 random images
images = list(IMAGE_DIR.glob("*.jpg"))
random.shuffle(images)

for image_path in images[:12]:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    if label_path.exists():

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # Convert YOLO coordinates to pixel coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            class_name = CLASS_NAMES.get(class_id, "Unknown")

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 20)),
                class_name,
                fill="red"
            )

    output_path = OUTPUT_DIR / image_path.name
    image.save(output_path)

    print(f"Saved: {output_path}")

print("\nVisualization complete!")
print(f"Images saved in: {OUTPUT_DIR}")