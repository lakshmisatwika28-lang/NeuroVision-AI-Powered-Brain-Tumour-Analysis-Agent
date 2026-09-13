import sys
import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# PATHS
# ============================================================

YOLO_MODEL_PATH = "models/yolo26_best.pt"
RESNET_MODEL_PATH = "models/resnet50_best.pth"

OUTPUT_DIR = "neurovision_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASSES
# ============================================================

YOLO_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

RESNET_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary"
}


# ============================================================
# INPUT
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print('python neurovision_pipeline.py "path_to_mri.jpg"')
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found.")
    sys.exit(1)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO26...")

yolo = YOLO(YOLO_MODEL_PATH)

print("Loading ResNet50...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

resnet = models.resnet50(weights=None)

resnet.fc = torch.nn.Linear(
    resnet.fc.in_features,
    3
)

checkpoint = torch.load(
    RESNET_MODEL_PATH,
    map_location=device
)

resnet.load_state_dict(
    checkpoint["model_state_dict"]
)

resnet.to(device)
resnet.eval()


# ============================================================
# IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not read image.")
    sys.exit(1)

original_image = image.copy()

height, width = image.shape[:2]


# ============================================================
# RESNET TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# OVERLAP RATIO
# ============================================================

def calculate_overlap_ratio(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    smaller_area = min(
        area1,
        area2
    )

    if smaller_area <= 0:
        return 0.0

    return intersection / smaller_area


# ============================================================
# CENTER DISTANCE
# ============================================================

def calculate_center_distance(box1, box2):

    center1_x = (
        box1[0] + box1[2]
    ) / 2

    center1_y = (
        box1[1] + box1[3]
    ) / 2

    center2_x = (
        box2[0] + box2[2]
    ) / 2

    center2_y = (
        box2[1] + box2[3]
    ) / 2

    return (
        (
            center1_x - center2_x
        ) ** 2
        +
        (
            center1_y - center2_y
        ) ** 2
    ) ** 0.5


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicate_detections(detections):

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    kept = []

    for detection in detections:

        is_duplicate = False

        for existing in kept:

            if (
                detection["class_id"]
                !=
                existing["class_id"]
            ):
                continue

            box1 = detection["box"]
            box2 = existing["box"]

            iou = calculate_iou(
                box1,
                box2
            )

            overlap_ratio = calculate_overlap_ratio(
                box1,
                box2
            )

            center_distance = calculate_center_distance(
                box1,
                box2
            )

            width1 = (
                box1[2] -
                box1[0]
            )

            height1 = (
                box1[3] -
                box1[1]
            )

            width2 = (
                box2[2] -
                box2[0]
            )

            height2 = (
                box2[3] -
                box2[1]
            )

            smaller_diagonal = min(
                (
                    width1 ** 2 +
                    height1 ** 2
                ) ** 0.5,

                (
                    width2 ** 2 +
                    height2 ** 2
                ) ** 0.5
            )

            very_close = (
                center_distance
                <=
                smaller_diagonal * 0.75
            )

            # ------------------------------------------------
            # DEBUG INFORMATION
            # ------------------------------------------------

            print()
            print(
                "Duplicate comparison:"
            )

            print(
                f"  Existing box: {box2}"
            )

            print(
                f"  New box:      {box1}"
            )

            print(
                f"  IoU:          {iou:.3f}"
            )

            print(
                f"  Small-box overlap: "
                f"{overlap_ratio:.3f}"
            )

            print(
                f"  Center distance: "
                f"{center_distance:.2f}px"
            )

            print(
                f"  Small-box diagonal: "
                f"{smaller_diagonal:.2f}px"
            )

            print(
                f"  Very close centers: "
                f"{very_close}"
            )


            # ------------------------------------------------
            # DUPLICATE RULE
            # ------------------------------------------------

            if (
                iou >= 0.30
                or
                overlap_ratio >= 0.50
                or
                (
                    very_close
                    and
                    overlap_ratio >= 0.20
                )
            ):

                print(
                    "  → DUPLICATE: removing lower-confidence box"
                )

                is_duplicate = True

                break

            else:

                print(
                    "  → KEPT: considered separate"
                )


        if not is_duplicate:

            kept.append(
                detection
            )

    return kept


# ============================================================
# CHARACTERISTICS
# ============================================================

def get_characteristics(box):

    x1, y1, x2, y2 = box

    tumor_width = x2 - x1
    tumor_height = y2 - y1

    area = (
        tumor_width *
        tumor_height
    )

    relative_area = (
        area /
        (width * height)
    ) * 100

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2


    if center_x < width / 3:
        horizontal = "Left"

    elif center_x < (
        2 * width / 3
    ):
        horizontal = "Center"

    else:
        horizontal = "Right"


    if center_y < height / 3:
        vertical = "Upper"

    elif center_y < (
        2 * height / 3
    ):
        vertical = "Middle"

    else:
        vertical = "Lower"


    location = (
        f"{vertical}-{horizontal}"
    )


    if relative_area < 1.5:
        area_category = "Small"

    elif relative_area <= 4.5:
        area_category = "Moderate"

    else:
        area_category = "Large"


    if location == "Middle-Center":
        location_category = "Central"

    else:
        location_category = "Peripheral"


    score = 0


    if area_category == "Moderate":
        score += 1

    elif area_category == "Large":
        score += 2


    if location_category == "Peripheral":
        score += 1


    if score <= 1:
        risk = "Lower"

    elif score == 2:
        risk = "Moderate"

    else:
        risk = "Higher"


    return {
        "width": tumor_width,
        "height": tumor_height,
        "relative_area": relative_area,
        "area_category": area_category,
        "location": location,
        "risk": risk,
        "risk_score": score
    }


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    crop,
    input_tensor,
    predicted_class
):

    target_layers = [
        resnet.layer4[-1]
    ]

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    with GradCAM(
        model=resnet,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]


    rgb_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )


    rgb_crop_224 = cv2.resize(
        rgb_crop,
        (224, 224)
    )


    rgb_crop_224 = (
        rgb_crop_224.astype(
            np.float32
        )
        /
        255.0
    )


    visualization = show_cam_on_image(
        rgb_crop_224,
        grayscale_cam,
        use_rgb=True
    )


    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )


    visualization = cv2.resize(
        visualization,
        (
            crop.shape[1],
            crop.shape[0]
        )
    )


    return visualization


# ============================================================
# HEADER
# ============================================================

print()

print(
    "======================================"
)

print(
    "        NEUROVISION ANALYSIS"
)

print(
    "======================================"
)

print()

print(
    "Image:",
    IMAGE_PATH
)

print(
    f"Image size: "
    f"{width} x {height}"
)


# ============================================================
# YOLO
# ============================================================

print()

print(
    "Running YOLO26..."
)


results = yolo.predict(
    source=image,
    conf=0.60,
    verbose=False
)


result = results[0]


raw_detections = []


for box in result.boxes:

    class_id = int(
        box.cls[0].item()
    )

    confidence = float(
        box.conf[0].item()
    )

    coordinates = (
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = map(
        int,
        coordinates
    )

    raw_detections.append({

        "class_id":
            class_id,

        "confidence":
            confidence,

        "box":
            [
                x1,
                y1,
                x2,
                y2
            ]
    })


print(
    f"Raw YOLO detections: "
    f"{len(raw_detections)}"
)


# ============================================================
# PRINT ALL RAW DETECTIONS
# ============================================================

print()

print(
    "RAW DETECTIONS"
)

print(
    "--------------------------------------"
)


for i, detection in enumerate(
    raw_detections,
    start=1
):

    print(
        f"Detection {i}: "
        f"{YOLO_CLASSES[detection['class_id']]}"
    )

    print(
        f"  Confidence: "
        f"{detection['confidence'] * 100:.2f}%"
    )

    print(
        f"  Bounding box: "
        f"{detection['box']}"
    )


# ============================================================
# IGNORE NO TUMOR
# ============================================================

tumor_detections = []


for detection in raw_detections:

    if detection["class_id"] == 2:
        continue

    tumor_detections.append(
        detection
    )


print()

print(
    f"Actual tumor detections: "
    f"{len(tumor_detections)}"
)


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

tumor_detections = (
    remove_duplicate_detections(
        tumor_detections
    )
)


print()

print(
    f"Detections after duplicate removal: "
    f"{len(tumor_detections)}"
)


# ============================================================
# NO TUMOR
# ============================================================

if len(tumor_detections) == 0:

    print()

    print(
        "--------------------------------------"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "--------------------------------------"
    )

    print()

    print(
        "No tumor detected by YOLO26."
    )

    print()

    print(
        "The remaining brain area was NOT "
        "classified as No Tumor."
    )


    base_name = os.path.splitext(
        os.path.basename(
            IMAGE_PATH
        )
    )[0]


    output_path = os.path.join(
        OUTPUT_DIR,
        base_name +
        "_final_analysis.jpg"
    )


    cv2.imwrite(
        output_path,
        original_image
    )


    print()

    print(
        "FINAL ANALYSIS IMAGE SAVED"
    )

    print(
        output_path
    )

    sys.exit(0)


# ============================================================
# FINAL IMAGE
# ============================================================

final_image = (
    original_image.copy()
)


# ============================================================
# PROCESS TUMORS
# ============================================================

for index, detection in enumerate(
    tumor_detections,
    start=1
):

    class_id = (
        detection["class_id"]
    )

    confidence = (
        detection["confidence"]
    )

    x1, y1, x2, y2 = (
        detection["box"]
    )


    tumor_class = (
        YOLO_CLASSES[
            class_id
        ]
    )


    print()

    print(
        "--------------------------------------"
    )

    print(
        f"Tumor #{index}"
    )

    print(
        "--------------------------------------"
    )


    print(
        f"YOLO class: "
        f"{tumor_class}"
    )


    print(
        f"YOLO confidence: "
        f"{confidence * 100:.2f}%"
    )


    print(
        f"Bounding box: "
        f"({x1}, {y1}) -> "
        f"({x2}, {y2})"
    )


    # ========================================================
    # CHARACTERISTICS
    # ========================================================

    characteristics = (
        get_characteristics(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )
    )


    print(
        f"Width: "
        f"{characteristics['width']:.1f} px"
    )


    print(
        f"Height: "
        f"{characteristics['height']:.1f} px"
    )


    print(
        f"Relative area: "
        f"{characteristics['relative_area']:.2f}%"
    )


    print(
        f"Area category: "
        f"{characteristics['area_category']}"
    )


    print(
        f"Location: "
        f"{characteristics['location']}"
    )


    print(
        f"Research risk: "
        f"{characteristics['risk']}"
    )


    print(
        f"Research risk score: "
        f"{characteristics['risk_score']}"
    )


    # ========================================================
    # CROP
    # ========================================================

    crop = original_image[
        max(0, y1):min(height, y2),
        max(0, x1):min(width, x2)
    ]


    if crop.size == 0:

        print(
            "WARNING: Empty crop. Skipping."
        )

        continue


    # ========================================================
    # RESNET
    # ========================================================

    input_tensor = transform(
        crop
    ).unsqueeze(
        0
    ).to(device)


    with torch.no_grad():

        outputs = resnet(
            input_tensor
        )


        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )
        )


        predicted_class_index = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )


        resnet_confidence = float(
            probabilities[
                0,
                predicted_class_index
            ].item()
        )


    predicted_class = (
        RESNET_CLASSES[
            predicted_class_index
        ]
    )


    print(
        f"ResNet50 prediction: "
        f"{predicted_class}"
    )


    print(
        f"ResNet50 confidence: "
        f"{resnet_confidence * 100:.2f}%"
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    try:

        gradcam_image = (
            generate_gradcam(
                crop,
                input_tensor,
                predicted_class_index
            )
        )


        blended_crop = cv2.addWeighted(
            crop,
            0.45,
            gradcam_image,
            0.55,
            0
        )


        final_image[
            max(0, y1):min(height, y2),
            max(0, x1):min(width, x2)
        ] = blended_crop


    except Exception as e:

        print(
            "Grad-CAM warning:",
            e
        )


    # ========================================================
    # BOUNDING BOX
    # ========================================================

    cv2.rectangle(
        final_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # ========================================================
    # LABEL
    # ========================================================

    label = (
        f"{predicted_class} | "
        f"YOLO {confidence * 100:.1f}% | "
        f"ResNet {resnet_confidence * 100:.1f}%"
    )


    cv2.putText(
        final_image,
        label,
        (
            x1,
            max(20, y1 - 10)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # RISK LABEL
    # ========================================================

    risk_label = (
        f"Research Risk: "
        f"{characteristics['risk']}"
    )


    cv2.putText(
        final_image,
        risk_label,
        (
            x1,
            min(
                height - 10,
                y2 + 18
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

base_name = os.path.splitext(
    os.path.basename(
        IMAGE_PATH
    )
)[0]


output_path = os.path.join(
    OUTPUT_DIR,
    base_name +
    "_final_analysis.jpg"
)


cv2.imwrite(
    output_path,
    final_image
)


# ============================================================
# FINAL
# ============================================================

print()

print(
    "======================================"
)

print(
    "FINAL ANALYSIS IMAGE SAVED"
)

print(
    "======================================"
)

print(
    output_path
)

print()

print(
    "Grad-CAM + tumor bounding boxes + "
    "predictions included."
)

print()

print(
    "No Tumor detections were ignored."
)