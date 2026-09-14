import os
import sys
import cv2
import torch
import numpy as np

from ultralytics import YOLO
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolo26_best.pt"
)

RESNET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_best.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "neurovision_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL CACHE
# ============================================================

yolo = None
resnet = None


def load_models():

    global yolo
    global resnet

    if yolo is None:

        print("Loading YOLO26...")

        yolo = YOLO(
            YOLO_MODEL_PATH
        )

    if resnet is None:

        print("Loading ResNet50...")

        resnet = models.resnet50(
            weights=None
        )

        resnet.fc = torch.nn.Linear(
            resnet.fc.in_features,
            3
        )

        checkpoint = torch.load(
            RESNET_MODEL_PATH,
            map_location=device
        )

        resnet.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        resnet.to(device)

        resnet.eval()


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# OVERLAP RATIO
# ============================================================

def calculate_overlap_ratio(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    smaller_area = min(
        area1,
        area2
    )

    if smaller_area <= 0:
        return 0.0

    return (
        intersection
        /
        smaller_area
    )


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
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicate_detections(
    detections
):

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

            overlap_ratio = (
                calculate_overlap_ratio(
                    box1,
                    box2
                )
            )

            center_distance = (
                calculate_center_distance(
                    box1,
                    box2
                )
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
                    width1 ** 2
                    +
                    height1 ** 2
                ) ** 0.5,

                (
                    width2 ** 2
                    +
                    height2 ** 2
                ) ** 0.5
            )

            very_close = (
                center_distance
                <=
                smaller_diagonal * 0.75
            )

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

                is_duplicate = True

                break

        if not is_duplicate:

            kept.append(
                detection
            )

    return kept


# ============================================================
# CHARACTERISTICS
# ============================================================

def get_characteristics(
    box,
    image_width,
    image_height
):

    x1, y1, x2, y2 = box

    tumor_width = (
        x2 - x1
    )

    tumor_height = (
        y2 - y1
    )

    area = (
        tumor_width
        *
        tumor_height
    )

    relative_area = (
        area
        /
        (
            image_width
            *
            image_height
        )
    ) * 100

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    if center_x < image_width / 3:

        horizontal = "Left"

    elif center_x < (
        2 * image_width / 3
    ):

        horizontal = "Center"

    else:

        horizontal = "Right"

    if center_y < image_height / 3:

        vertical = "Upper"

    elif center_y < (
        2 * image_height / 3
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

        "relative_area":
            relative_area,

        "area_category":
            area_category,

        "location":
            location,

        "location_category":
            location_category,

        "risk":
            risk,

        "risk_score":
            score
    }


# ============================================================
# RESNET TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


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
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_image(
    image_path
):

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    load_models()

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Could not read image."
        )

    original_image = (
        image.copy()
    )

    height, width = (
        image.shape[:2]
    )

    # ========================================================
    # YOLO
    # ========================================================

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

            "class_name":
                YOLO_CLASSES[
                    class_id
                ],

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

    # ========================================================
    # IGNORE NO TUMOR
    # ========================================================

    tumor_detections = [

        detection

        for detection
        in raw_detections

        if detection[
            "class_id"
        ] != 2
    ]

    tumor_detections = (
        remove_duplicate_detections(
            tumor_detections
        )
    )

    # ========================================================
    # FINAL IMAGE
    # ========================================================

    final_image = (
        original_image.copy()
    )

    processed_detections = []

    # ========================================================
    # NO TUMOR
    # ========================================================

    if len(
        tumor_detections
    ) == 0:

        base_name = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            base_name
            +
            "_final_analysis.jpg"
        )

        cv2.imwrite(
            output_path,
            final_image
        )

        return {

            "image_path":
                image_path,

            "image_width":
                width,

            "image_height":
                height,

            "detections": [],

            "tumor_detected":
                False,

            "result_image":
                output_path,

            "message":
                "No tumor detected by YOLO26."
        }

    # ========================================================
    # PROCESS TUMORS
    # ========================================================

    for index, detection in enumerate(
        tumor_detections,
        start=1
    ):

        class_id = (
            detection["class_id"]
        )

        yolo_confidence = (
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

        characteristics = (
            get_characteristics(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                width,
                height
            )
        )

        crop = original_image[
            max(0, y1):
            min(height, y2),

            max(0, x1):
            min(width, x2)
        ]

        if crop.size == 0:

            continue

        # ====================================================
        # RESNET
        # ====================================================

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

        # ====================================================
        # GRAD-CAM
        # ====================================================

        gradcam_path = None

        try:

            gradcam_image = (
                generate_gradcam(
                    crop,
                    input_tensor,
                    predicted_class_index
                )
            )

            blended_crop = (
                cv2.addWeighted(
                    crop,
                    0.45,
                    gradcam_image,
                    0.55,
                    0
                )
            )

            final_image[
                max(0, y1):
                min(height, y2),

                max(0, x1):
                min(width, x2)
            ] = blended_crop

        except Exception as e:

            print(
                "Grad-CAM warning:",
                e
            )

        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            final_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ====================================================
        # LABEL
        # ====================================================

        label = (
            f"{predicted_class} | "
            f"YOLO "
            f"{yolo_confidence * 100:.1f}% | "
            f"ResNet "
            f"{resnet_confidence * 100:.1f}%"
        )

        cv2.putText(
            final_image,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

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

        # ====================================================
        # STORE RESULT
        # ====================================================

        processed_detections.append({

            "id":
                index,

            "yolo_class":
                tumor_class,

            "yolo_class_id":
                class_id,

            "yolo_confidence":
                yolo_confidence,

            "bounding_box": {

                "x1": x1,

                "y1": y1,

                "x2": x2,

                "y2": y2
            },

            "resnet_prediction":
                predicted_class,

            "resnet_class_id":
                predicted_class_index,

            "resnet_confidence":
                resnet_confidence,

            "tumor_characteristics":
                characteristics,

            "model_agreement":
                tumor_class
                ==
                predicted_class
        })

    # ========================================================
    # SAVE RESULT IMAGE
    # ========================================================

    base_name = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        base_name
        +
        "_final_analysis.jpg"
    )

    cv2.imwrite(
        output_path,
        final_image
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    agreement_count = sum(

        1

        for detection
        in processed_detections

        if detection[
            "model_agreement"
        ]
    )

    total_count = len(
        processed_detections
    )

    return {

        "image_path":
            image_path,

        "image_width":
            width,

        "image_height":
            height,

        "tumor_detected":
            True,

        "detections":
            processed_detections,

        "detection_count":
            total_count,

        "model_agreement_count":
            agreement_count,

        "model_agreement":
            (
                agreement_count
                ==
                total_count
            ),

        "result_image":
            output_path
    }


# ============================================================
# COMMAND LINE MODE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            'python neurovision_pipeline.py "path_to_mri.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

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
        image_path
    )

    try:

        result = analyze_image(
            image_path
        )

        print()

        print(
            "FINAL ANALYSIS"
        )

        print(
            "--------------------------------------"
        )

        if not result[
            "tumor_detected"
        ]:

            print(
                "No tumor detected by YOLO26."
            )

        else:

            print(
                f"Tumor detections: "
                f"{result['detection_count']}"
            )

            for detection in result[
                "detections"
            ]:

                print()

                print(
                    f"Tumor #{detection['id']}"
                )

                print(
                    f"YOLO class: "
                    f"{detection['yolo_class']}"
                )

                print(
                    f"YOLO confidence: "
                    f"{detection['yolo_confidence'] * 100:.2f}%"
                )

                print(
                    f"ResNet50 prediction: "
                    f"{detection['resnet_prediction']}"
                )

                print(
                    f"ResNet50 confidence: "
                    f"{detection['resnet_confidence'] * 100:.2f}%"
                )

                characteristics = (
                    detection[
                        "tumor_characteristics"
                    ]
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
                    f"Model agreement: "
                    f"{detection['model_agreement']}"
                )

        print()

        print(
            "FINAL ANALYSIS IMAGE SAVED"
        )

        print(
            result["result_image"]
        )

        print()

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        sys.exit(1)