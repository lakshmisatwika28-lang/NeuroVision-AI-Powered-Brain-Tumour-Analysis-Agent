from ultralytics import YOLO
from pathlib import Path
from PIL import Image

MODEL_PATH = "models/yolo26_best.pt"
DATASET_DIR = Path("dataset/NeuroVision_YOLO_v2")
OUTPUT_DIR = Path("tumor_regions")

model = YOLO(MODEL_PATH)

class_names = {
    0: "Glioma",
    1: "Meningioma",
    2: "No_Tumor",
    3: "Pituitary"
}

for split in ["train", "val", "test"]:
    image_dir = DATASET_DIR / "images" / split

    for class_name in class_names.values():
        (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.jpg"))

    print(f"\nProcessing {split}: {len(images)} images")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")

        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for i, (box, cls) in enumerate(
            zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy()
            )
        ):
            class_id = int(cls)

            if class_id not in class_names:
                continue

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            region = image.crop((x1, y1, x2, y2))

            class_name = class_names[class_id]

            output_file = (
                OUTPUT_DIR
                / split
                / class_name
                / f"{image_path.stem}_region_{i}.jpg"
            )

            region.save(output_file)

    print(f"Finished {split}")

print("\nTUMOUR REGION EXTRACTION COMPLETE!")