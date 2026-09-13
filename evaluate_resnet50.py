import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# SETTINGS
# ==============================

TEST_DIR = "classifier_regions/test"
MODEL_PATH = "models/resnet50_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==============================
# TEST TRANSFORMS
# ==============================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# TEST DATASET
# ==============================

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


# ==============================
# CREATE RESNET50
# ==============================

print("\nLoading ResNet50...")

model = models.resnet50(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)


# ==============================
# LOAD TRAINED CHECKPOINT
# ==============================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ==============================
# TEST EVALUATION
# ==============================

all_predictions = []
all_labels = []

print("\nRunning test evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==============================
# METRICS
# ==============================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)


# ==============================
# RESULTS
# ==============================

print("\n")
print("=" * 50)
print("RESNET50 TEST RESULTS")
print("=" * 50)

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")


# ==============================
# CLASSIFICATION REPORT
# ==============================

class_names = test_dataset.classes

print("\nPER-CLASS RESULTS")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=2
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nCONFUSION MATRIX")
print(cm)

print("\nRows = Actual")
print("Columns = Predicted")

print("\nRESNET50 TEST EVALUATION COMPLETE!")