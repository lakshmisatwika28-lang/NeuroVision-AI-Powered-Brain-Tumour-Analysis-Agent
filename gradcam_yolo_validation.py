import os
import re
import csv
import random
import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet50_best.pth"

CLASSIFIER_ROOT = "classifier_regions/test"

YOLO_IMAGE_ROOT = "dataset/NeuroVision_YOLO_v2/images/test"
YOLO_LABEL_ROOT = "dataset/NeuroVision_YOLO_v2/labels/test"

OUTPUT_DIR = "gradcam_yolo_validation"

SAMPLES_PER_CLASS = 5

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("NEUROVISION GRAD-CAM + YOLO VALIDATION")
print("=" * 65)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD RESNET50
# ============================================================

print("\nLoading ResNet50...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("ResNet50 loaded successfully.")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FIND ORIGINAL YOLO IMAGE
# ============================================================

def find_original_image(crop_path):

    filename = os.path.basename(crop_path)

    # Example:
    #
    # train_glioma_gg (126)_crop1.jpg
    #
    # becomes:
    #
    # train_glioma_gg (126).jpg

    match = re.match(
        r"(.+)_crop\d+$",
        os.path.splitext(filename)[0]
    )

    if match is None:
        return None

    original_name = match.group(1) + ".jpg"

    original_path = os.path.join(
        YOLO_IMAGE_ROOT,
        original_name
    )

    if os.path.exists(original_path):
        return original_path

    return None


# ============================================================
# READ YOLO LABELS
# ============================================================

def read_yolo_labels(image_path):

    filename = os.path.basename(image_path)

    label_name = (
        os.path.splitext(filename)[0]
        + ".txt"
    )

    label_path = os.path.join(
        YOLO_LABEL_ROOT,
        label_name
    )

    if not os.path.exists(label_path):
        return []

    image = cv2.imread(image_path)

    if image is None:
        return []

    height, width = image.shape[:2]

    boxes = []

    with open(label_path, "r") as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])

            box_width = float(parts[3])
            box_height = float(parts[4])

            x1 = int(
                (x_center - box_width / 2)
                * width
            )

            y1 = int(
                (y_center - box_height / 2)
                * height
            )

            x2 = int(
                (x_center + box_width / 2)
                * width
            )

            y2 = int(
                (y_center + box_height / 2)
                * height
            )

            boxes.append({
                "class_id": class_id,
                "x1": max(0, x1),
                "y1": max(0, y1),
                "x2": min(width - 1, x2),
                "y2": min(height - 1, y2)
            })

    return boxes


# ============================================================
# GET CROP NUMBER
# ============================================================

def get_crop_number(crop_path):

    filename = os.path.basename(crop_path)

    match = re.search(
        r"_crop(\d+)",
        filename
    )

    if match is None:
        return None

    return int(match.group(1))


# ============================================================
# FIND EXACT YOLO BOX USED FOR THIS CROP
# ============================================================

def find_crop_box(
    crop_path,
    original_image_path
):

    boxes = read_yolo_labels(
        original_image_path
    )

    if not boxes:
        return None

    crop_number = get_crop_number(
        crop_path
    )

    if crop_number is None:
        return None

    # The crop number corresponds to the
    # annotation order used while creating
    # classifier_regions.

    tumor_boxes = [
        box
        for box in boxes
        if box["class_id"] in [0, 1, 3]
    ]

    if crop_number >= len(tumor_boxes):
        return None

    return tumor_boxes[crop_number]


# ============================================================
# CALCULATE CAM ACTIVATION INSIDE CROP
# ============================================================

def calculate_activation_statistics(
    grayscale_cam
):

    # Top 30% strongest activations

    threshold = np.percentile(
        grayscale_cam,
        70
    )

    activation_mask = (
        grayscale_cam >= threshold
    )

    total_pixels = (
        activation_mask.size
    )

    active_pixels = (
        np.sum(activation_mask)
    )

    activation_percentage = (
        active_pixels /
        total_pixels
    ) * 100

    # Since the classifier receives the
    # tumor crop itself, we evaluate how
    # concentrated the activation is
    # rather than comparing to the full
    # image coordinates.

    center_y, center_x = np.array(
        grayscale_cam.shape
    ) / 2

    y_indices, x_indices = np.where(
        activation_mask
    )

    if len(x_indices) > 0:

        distances = np.sqrt(
            (x_indices - center_x) ** 2
            +
            (y_indices - center_y) ** 2
        )

        mean_distance = np.mean(
            distances
        )

    else:

        mean_distance = 0

    return (
        activation_percentage,
        mean_distance
    )


# ============================================================
# SELECT IMAGES
# ============================================================

selected_images = []

random.seed(42)

for class_name in CLASS_NAMES:

    class_dir = os.path.join(
        CLASSIFIER_ROOT,
        class_name
    )

    images = [
        os.path.join(
            class_dir,
            filename
        )
        for filename in os.listdir(
            class_dir
        )
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(images)

    selected = images[
        :SAMPLES_PER_CLASS
    ]

    for image_path in selected:

        selected_images.append(
            (
                class_name,
                image_path
            )
        )


print(
    "\nSelected",
    len(selected_images),
    "images."
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

for class_name in CLASS_NAMES:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            class_name
        ),
        exist_ok=True
    )


# ============================================================
# GRAD-CAM
# ============================================================

target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)

csv_file = open(
    csv_path,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Image",
    "Actual Class",
    "Predicted Class",
    "Confidence (%)",
    "Correct",
    "Original MRI Found",
    "YOLO Box Found",
    "CAM Active Area (%)",
    "CAM Mean Center Distance"
])


# ============================================================
# PROCESS IMAGES
# ============================================================

correct_predictions = 0

for index, (
    actual_class,
    crop_path
) in enumerate(
    selected_images,
    start=1
):

    print("\n" + "-" * 65)

    print(
        f"[{index}/{len(selected_images)}]",
        os.path.basename(crop_path)
    )

    # --------------------------------------------------------
    # Load crop
    # --------------------------------------------------------

    image = Image.open(
        crop_path
    ).convert("RGB")

    image_np = np.array(image)

    display_image = (
        image_np.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[
                0,
                predicted_index
            ].item()
            * 100
        )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    correct = (
        predicted_class
        ==
        actual_class
    )

    if correct:
        correct_predictions += 1

    print(
        "Actual:",
        actual_class
    )

    print(
        "Predicted:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    # --------------------------------------------------------
    # Find original MRI
    # --------------------------------------------------------

    original_path = find_original_image(
        crop_path
    )

    original_found = (
        original_path is not None
    )

    if original_found:

        print(
            "Original MRI: FOUND"
        )

    else:

        print(
            "Original MRI: NOT FOUND"
        )

    # --------------------------------------------------------
    # Find YOLO box
    # --------------------------------------------------------

    crop_box = None

    if original_found:

        crop_box = find_crop_box(
            crop_path,
            original_path
        )

    box_found = (
        crop_box is not None
    )

    if box_found:

        print(
            "YOLO tumor box: FOUND"
        )

    else:

        print(
            "YOLO tumor box: NOT FOUND"
        )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(
            predicted_index
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # --------------------------------------------------------
    # CAM statistics
    # --------------------------------------------------------

    (
        active_area,
        center_distance
    ) = calculate_activation_statistics(
        grayscale_cam
    )

    print(
        "CAM active area:",
        f"{active_area:.2f}%"
    )

    print(
        "CAM mean center distance:",
        f"{center_distance:.2f}"
    )

    # --------------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------------

    cam_overlay = show_cam_on_image(
        display_image,
        grayscale_cam,
        use_rgb=True
    )

    result = cv2.cvtColor(
        cam_overlay,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Add information
    # --------------------------------------------------------

    cv2.putText(
        result,
        f"Actual: {actual_class}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Pred: {predicted_class} "
        f"({confidence:.2f}%)",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    output_filename = (
        os.path.splitext(
            os.path.basename(crop_path)
        )[0]
        + "_validation.jpg"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        actual_class,
        output_filename
    )

    cv2.imwrite(
        output_path,
        result
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    writer.writerow([
        os.path.basename(crop_path),
        actual_class,
        predicted_class,
        f"{confidence:.2f}",
        correct,
        original_found,
        box_found,
        f"{active_area:.2f}",
        f"{center_distance:.2f}"
    ])


# ============================================================
# FINISH
# ============================================================

csv_file.close()

accuracy = (
    correct_predictions /
    len(selected_images)
) * 100

print("\n" + "=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)

print(
    "Correct predictions:",
    f"{correct_predictions}/{len(selected_images)}"
)

print(
    "Sample accuracy:",
    f"{accuracy:.2f}%"
)

print(
    "\nResults folder:",
    OUTPUT_DIR
)

print(
    "CSV:",
    csv_path
)

print("=" * 65)