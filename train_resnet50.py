import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================
# SETTINGS
# =========================

TRAIN_DIR = "classifier_regions/train"
VAL_DIR = "classifier_regions/val"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

MODEL_SAVE_PATH = "models/resnet50_best.pth"


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if device.type == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    print("CPU threads:", torch.get_num_threads())


# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASETS
# =========================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClasses:")
print(train_dataset.class_to_idx)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# =========================
# RESNET50
# =========================

print("\nLoading pretrained ResNet50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

# Replace ImageNet's 1000-class output
# with our 3 tumor classes.
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    3
)

model = model.to(device)


# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================

best_val_accuracy = 0.0

print("\nStarting ResNet50 training...\n")

total_start = time.time()


for epoch in range(EPOCHS):

    epoch_start = time.time()

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss_total / val_total
    val_accuracy = 100 * val_correct / val_total


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "class_to_idx": train_dataset.class_to_idx,
            "image_size": IMAGE_SIZE,
            "num_classes": 3
        }, MODEL_SAVE_PATH)

        best_marker = "  <-- BEST MODEL"

    else:
        best_marker = ""


    epoch_time = time.time() - epoch_start

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}% "
        f"| Time: {epoch_time / 60:.1f} min"
        f"{best_marker}"
    )


# =========================
# FINAL
# =========================

total_time = time.time() - total_start

print("\n========================================")
print("RESNET50 TRAINING COMPLETE!")
print("========================================")

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

print(f"Total Training Time: {total_time / 60:.1f} minutes")

print("\nBest model saved to:")
print(MODEL_SAVE_PATH)