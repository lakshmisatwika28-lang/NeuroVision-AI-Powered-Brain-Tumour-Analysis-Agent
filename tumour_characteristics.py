import os
import csv

# =========================
# PATHS
# =========================

IMAGE_DIR = "dataset/NeuroVision_YOLO_v2/images"
LABEL_DIR = "dataset/NeuroVision_YOLO_v2/labels"

OUTPUT_FILE = "tumor_characteristics.csv"

CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

# =========================
# PROCESS DATASET
# =========================

results = []

for split in ["train", "val", "test"]:

    image_folder = os.path.join(IMAGE_DIR, split)
    label_folder = os.path.join(LABEL_DIR, split)

    if not os.path.exists(image_folder):
        continue

    for filename in os.listdir(image_folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_folder, filename)

        # Get image dimensions
        from PIL import Image

        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except Exception:
            continue

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_folder, label_filename)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        tumor_index = 0

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # Convert normalized coordinates to pixels
            width_px = box_width * image_width
            height_px = box_height * image_height

            area_px = width_px * height_px

            image_area = image_width * image_height

            relative_area = (
                area_px / image_area
            ) * 100

            center_x_px = x_center * image_width
            center_y_px = y_center * image_height

            # -------------------------
            # Horizontal location
            # -------------------------

            if x_center < 0.33:
                horizontal = "Left"
            elif x_center < 0.67:
                horizontal = "Center"
            else:
                horizontal = "Right"

            # -------------------------
            # Vertical location
            # -------------------------

            if y_center < 0.33:
                vertical = "Upper"
            elif y_center < 0.67:
                vertical = "Middle"
            else:
                vertical = "Lower"

            location = f"{vertical}-{horizontal}"

            results.append({
                "split": split,
                "image": filename,
                "tumor_index": tumor_index,
                "tumor_class": CLASSES.get(
                    class_id,
                    "Unknown"
                ),
                "image_width": image_width,
                "image_height": image_height,
                "box_width_px": round(width_px, 2),
                "box_height_px": round(height_px, 2),
                "box_area_px": round(area_px, 2),
                "relative_area_percent": round(
                    relative_area,
                    4
                ),
                "center_x_px": round(
                    center_x_px,
                    2
                ),
                "center_y_px": round(
                    center_y_px,
                    2
                ),
                "horizontal_location": horizontal,
                "vertical_location": vertical,
                "location": location
            })

            tumor_index += 1


# =========================
# SAVE CSV
# =========================

if results:

    fieldnames = results[0].keys()

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n===================================")
    print("TUMOR CHARACTERISTICS COMPLETE")
    print("===================================")

    print("Total tumor annotations:", len(results))
    print("Saved:", OUTPUT_FILE)

else:

    print("No tumor annotations found.")