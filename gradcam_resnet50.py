import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_PATH = "models/resnet50_best.pth"
TEST_DIR = "classifier_regions/test"
OUTPUT_DIR = "gradcam_results"

NUM_IMAGES_PER_CLASS = 5

CLASSES = ["Glioma", "Meningioma", "Pituitary"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# Load checkpoint
# -----------------------------

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


# -----------------------------
# Image transformation
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Grad-CAM target layer
# -----------------------------

target_layers = [model.layer4[-1]]


# -----------------------------
# Create output directory
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Process images
# -----------------------------

for class_name in CLASSES:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.seed(42)
    random.shuffle(images)

    selected_images = images[:NUM_IMAGES_PER_CLASS]

    print("\n" + "=" * 60)
    print("CLASS:", class_name)
    print("Images selected:", len(selected_images))
    print("=" * 60)

    class_output_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        class_output_dir,
        exist_ok=True
    )

    for image_name in selected_images:

        image_path = os.path.join(
            class_dir,
            image_name
        )

        print("\nProcessing:", image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Original image resized for visualization
        rgb_image = image.resize((224, 224))

        rgb_array = (
            torch.tensor(
                list(rgb_image.getdata()),
                dtype=torch.float32
            )
            .reshape(224, 224, 3)
            .numpy()
            / 255.0
        )

        # Model input
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():

            output = model(input_tensor)

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
            ].item() * 100

        predicted_name = CLASSES[predicted_class]

        print(
            f"Prediction : {predicted_name}"
        )

        print(
            f"Confidence : {confidence:.2f}%"
        )

        # Grad-CAM
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
            input_tensor=input_tensor,
            targets=targets
        )[0]

        visualization = show_cam_on_image(
            rgb_array,
            grayscale_cam,
            use_rgb=True
        )

        # Output filename
        base_name = os.path.splitext(
            image_name
        )[0]

        output_name = (
            f"{base_name}"
            f"_pred_{predicted_name}"
            f"_conf_{confidence:.1f}"
            f"_gradcam.jpg"
        )

        output_path = os.path.join(
            class_output_dir,
            output_name
        )

        Image.fromarray(
            visualization
        ).save(output_path)

        print(
            "Saved:",
            output_path
        )


print("\n" + "=" * 60)
print("BATCH GRAD-CAM COMPLETE")
print("=" * 60)
print("Results saved in:", OUTPUT_DIR)