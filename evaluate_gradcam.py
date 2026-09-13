import os
import random

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/resnet50_best.pth"
TEST_DIR = "classifier_regions/test"

NUM_IMAGES_PER_CLASS = 5

CLASSES = [
    "Glioma",
    "Meningioma",
    "Pituitary"
]

IMAGE_SIZE = 224

MASK_PERCENTAGE = 0.20

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet50...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

target_layers = [
    model.layer4[-1]
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict(image_tensor):

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return predicted_class, confidence


def create_masked_tensor(
    image_tensor,
    cam,
    percentage=0.20
):

    masked = image_tensor.clone()

    flat_cam = cam.flatten()

    number_pixels = int(
        len(flat_cam) * percentage
    )

    threshold_index = (
        len(flat_cam) - number_pixels
    )

    threshold = torch.tensor(
        np.sort(flat_cam)[threshold_index]
    )

    mask = torch.tensor(
        cam >= threshold.item(),
        dtype=torch.bool
    )

    mask = mask.unsqueeze(0).unsqueeze(0)

    mask = mask.float()

    mask = torch.nn.functional.interpolate(
        mask,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest"
    )

    # Replace important region with mean-normalized image
    masked = (
        masked * (1 - mask)
    )

    return masked


def create_random_mask(
    image_tensor,
    percentage=0.20
):

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    total_pixels = height * width

    number_pixels = int(
        total_pixels * percentage
    )

    indices = random.sample(
        range(total_pixels),
        number_pixels
    )

    mask = torch.zeros(
        height * width
    )

    mask[indices] = 1

    mask = mask.reshape(
        height,
        width
    )

    mask = mask.unsqueeze(0).unsqueeze(0)

    mask = mask.to(device)

    masked = image_tensor.clone()

    masked = (
        masked * (1 - mask)
    )

    return masked


# ============================================================
# RESULTS
# ============================================================

results = []

random.seed(42)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in CLASSES:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    images = [
        f
        for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(images)

    selected_images = images[
        :NUM_IMAGES_PER_CLASS
    ]

    print("\n" + "=" * 65)
    print("CLASS:", class_name)
    print("=" * 65)

    for image_name in selected_images:

        image_path = os.path.join(
            class_dir,
            image_name
        )

        print("\nImage:", image_name)

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # ORIGINAL PREDICTION
        # ----------------------------------------------------

        predicted_class, original_confidence = predict(
            image_tensor
        )

        predicted_name = CLASSES[
            predicted_class
        ]

        correct = (
            predicted_name == class_name
        )

        print(
            f"Prediction : {predicted_name}"
        )

        print(
            f"Original confidence : "
            f"{original_confidence * 100:.2f}%"
        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets
        )[0]

        # ----------------------------------------------------
        # CAM-MASKED IMAGE
        # ----------------------------------------------------

        cam_masked = create_masked_tensor(
            image_tensor,
            grayscale_cam,
            MASK_PERCENTAGE
        )

        _, cam_confidence = predict(
            cam_masked
        )

        # ----------------------------------------------------
        # RANDOM-MASKED IMAGE
        # ----------------------------------------------------

        random_masked = create_random_mask(
            image_tensor,
            MASK_PERCENTAGE
        )

        _, random_confidence = predict(
            random_masked
        )

        # ----------------------------------------------------
        # CONFIDENCE DROPS
        # ----------------------------------------------------

        cam_drop = (
            original_confidence
            - cam_confidence
        )

        random_drop = (
            original_confidence
            - random_confidence
        )

        print(
            f"CAM-masked confidence : "
            f"{cam_confidence * 100:.2f}%"
        )

        print(
            f"Random-masked confidence : "
            f"{random_confidence * 100:.2f}%"
        )

        print(
            f"CAM confidence drop : "
            f"{cam_drop * 100:.2f} percentage points"
        )

        print(
            f"Random confidence drop : "
            f"{random_drop * 100:.2f} percentage points"
        )

        results.append({
            "actual": class_name,
            "predicted": predicted_name,
            "correct": correct,
            "original": original_confidence,
            "cam_confidence": cam_confidence,
            "random_confidence": random_confidence,
            "cam_drop": cam_drop,
            "random_drop": random_drop
        })


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("GRAD-CAM FAITHFULNESS EVALUATION")
print("=" * 65)

total = len(results)

correct_predictions = sum(
    r["correct"]
    for r in results
)

average_original = (
    sum(r["original"] for r in results)
    / total
)

average_cam_drop = (
    sum(r["cam_drop"] for r in results)
    / total
)

average_random_drop = (
    sum(r["random_drop"] for r in results)
    / total
)

print(
    f"\nImages tested          : {total}"
)

print(
    f"Correct predictions    : "
    f"{correct_predictions}/{total}"
)

print(
    f"Average confidence     : "
    f"{average_original * 100:.2f}%"
)

print(
    f"CAM confidence drop    : "
    f"{average_cam_drop * 100:.2f} percentage points"
)

print(
    f"Random confidence drop : "
    f"{average_random_drop * 100:.2f} percentage points"
)

difference = (
    average_cam_drop
    - average_random_drop
)

print(
    f"CAM vs random difference: "
    f"{difference * 100:.2f} percentage points"
)

print("\n" + "=" * 65)

if average_cam_drop > average_random_drop:

    print(
        "RESULT: Grad-CAM regions have stronger "
        "influence than random regions."
    )

else:

    print(
        "RESULT: Grad-CAM did not show stronger "
        "influence than random regions."
    )

print("=" * 65)