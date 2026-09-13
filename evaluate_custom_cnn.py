import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from models.custom_cnn import CustomCNN


# =========================
# SETTINGS
# =========================

TEST_DIR = "classifier_regions/test"
MODEL_PATH = "models/custom_cnn_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================
# TRANSFORM
# =========================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# TEST DATASET
# =========================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nClasses:")
print(test_dataset.class_to_idx)

print("\nTest images:", len(test_dataset))


# =========================
# LOAD MODEL
# =========================

model = CustomCNN(num_classes=3)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()


# =========================
# PREDICTIONS
# =========================

all_labels = []
all_predictions = []

print("\nRunning test evaluation...\n")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# =========================
# RESULTS
# =========================

print("========================================")
print("CUSTOM CNN TEST RESULTS")
print("========================================")

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")


# =========================
# CLASSIFICATION REPORT
# =========================

class_names = list(test_dataset.class_to_idx.keys())

print("\n========================================")
print("PER-CLASS RESULTS")
print("========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)

print("\nRows = Actual")
print("Columns = Predicted")