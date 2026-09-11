from pathlib import Path
import random
import shutil
from collections import Counter

# ============================================================
# SETTINGS
# ============================================================

SOURCE_ROOT = Path("dataset/NeuroVision_YOLO")
OUTPUT_ROOT = Path("dataset/NeuroVision_YOLO_v2")

TEST_RATIO = 0.10
RANDOM_SEED = 42

CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

random.seed(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# COPY EXISTING VALIDATION SET
# ============================================================

print("\nCopying validation set...")

val_images = list((SOURCE_ROOT / "images" / "val").glob("*.jpg"))

for image_path in val_images:
    label_path = SOURCE_ROOT / "labels" / "val" / f"{image_path.stem}.txt"

    shutil.copy2(
        image_path,
        OUTPUT_ROOT / "images" / "val" / image_path.name
    )

    # Copy label if it exists.
    # Empty labels are allowed for negative examples.
    if label_path.exists():
        shutil.copy2(
            label_path,
            OUTPUT_ROOT / "labels" / "val" / label_path.name
        )

print(f"Validation images copied: {len(val_images)}")


# ============================================================
# GET TRAINING IMAGES
# ============================================================

train_source = SOURCE_ROOT / "images" / "train"
train_label_source = SOURCE_ROOT / "labels" / "train"

all_train_images = list(train_source.glob("*.jpg"))

print(f"Original training images: {len(all_train_images)}")


# ============================================================
# GROUP IMAGES BY THEIR ORIGINAL CLASS/FOLDER
#
# The prepared dataset filenames contain the original class,
# e.g. train_glioma_...
#
# We use the label files to determine the classes actually
# represented by the annotations.
# ============================================================

groups = {}

for image_path in all_train_images:

    label_path = train_label_source / f"{image_path.stem}.txt"

    classes_in_image = set()

    if label_path.exists():

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))

                if class_id in CLASSES:
                    classes_in_image.add(class_id)

    # Images with no annotation are treated separately.
    if not classes_in_image:
        group = "NoAnnotation"
    else:
        # An image can contain multiple classes.
        # Represent the combination so the entire image stays together.
        group = "_".join(map(str, sorted(classes_in_image)))

    groups.setdefault(group, []).append(image_path)


# ============================================================
# SPLIT EACH GROUP
# ============================================================

test_images = []
remaining_train_images = []

print("\nGroups found:")

for group, images in sorted(groups.items()):

    random.shuffle(images)

    test_count = max(1, round(len(images) * TEST_RATIO))

    group_test = images[:test_count]
    group_train = images[test_count:]

    test_images.extend(group_test)
    remaining_train_images.extend(group_train)

    print(
        f"Group {group:15s} | "
        f"Total: {len(images):4d} | "
        f"Train: {len(group_train):4d} | "
        f"Test: {len(group_test):4d}"
    )


# ============================================================
# COPY TRAINING IMAGES
# ============================================================

print("\nCopying training images...")

for image_path in remaining_train_images:

    label_path = train_label_source / f"{image_path.stem}.txt"

    shutil.copy2(
        image_path,
        OUTPUT_ROOT / "images" / "train" / image_path.name
    )

    if label_path.exists():
        shutil.copy2(
            label_path,
            OUTPUT_ROOT / "labels" / "train" / label_path.name
        )


# ============================================================
# COPY TEST IMAGES
# ============================================================

print("Copying test images...")

for image_path in test_images:

    label_path = train_label_source / f"{image_path.stem}.txt"

    shutil.copy2(
        image_path,
        OUTPUT_ROOT / "images" / "test" / image_path.name
    )

    if label_path.exists():
        shutil.copy2(
            label_path,
            OUTPUT_ROOT / "labels" / "test" / label_path.name
        )


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_content = """path: .

train: images/train
val: images/val
test: images/test

names:
  0: Glioma
  1: Meningioma
  2: No Tumor
  3: Pituitary
"""

with open(OUTPUT_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)


# ============================================================
# VERIFY NO IMAGE OVERLAP
# ============================================================

train_names = {
    p.name for p in
    (OUTPUT_ROOT / "images" / "train").glob("*.jpg")
}

val_names = {
    p.name for p in
    (OUTPUT_ROOT / "images" / "val").glob("*.jpg")
}

test_names = {
    p.name for p in
    (OUTPUT_ROOT / "images" / "test").glob("*.jpg")
}

train_val_overlap = train_names & val_names
train_test_overlap = train_names & test_names
val_test_overlap = val_names & test_names


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("NEUROVISION TEST SPLIT COMPLETE")
print("=" * 60)

print(f"\nTrain images : {len(train_names)}")
print(f"Val images   : {len(val_names)}")
print(f"Test images  : {len(test_names)}")
print(f"Total images : {len(train_names) + len(val_names) + len(test_names)}")

print("\nOverlap check:")
print(f"Train ↔ Val  : {len(train_val_overlap)}")
print(f"Train ↔ Test : {len(train_test_overlap)}")
print(f"Val ↔ Test   : {len(val_test_overlap)}")

if train_val_overlap or train_test_overlap or val_test_overlap:
    print("\n❌ WARNING: IMAGE OVERLAP DETECTED!")
else:
    print("\n✅ No image overlap detected.")


# ============================================================
# VERIFY IMAGE/LABEL PAIRS
# ============================================================

print("\nChecking image-label pairs...")

problems = []

for split in ["train", "val", "test"]:

    image_dir = OUTPUT_ROOT / "images" / split
    label_dir = OUTPUT_ROOT / "labels" / split

    for image_path in image_dir.glob("*.jpg"):

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            problems.append(
                f"{split}: missing label for {image_path.name}"
            )

print(f"Pairing problems: {len(problems)}")

if problems:
    for problem in problems[:20]:
        print(problem)
else:
    print("✅ All images have corresponding label files.")


print("\n" + "=" * 60)
print("READY FOR YOLO26 TRAINING")
print("=" * 60)