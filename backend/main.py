import os
import shutil
import uuid
from datetime import datetime

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

import sys


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)

from neurovision_pipeline import analyze_image


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "uploads"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "neurovision_results"
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "reports"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# STORE ANALYSES
# ============================================================

ANALYSIS_STORE = {}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="NeuroVision API",
    description="NeuroVision Brain MRI Analysis Backend",
    version="1.0.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)

app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "NeuroVision API"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# ANALYZE MRI
# ============================================================

@app.post("/api/analyze")
async def analyze_mri(
    request: Request,
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, BMP or WEBP."
            )
        )

    unique_name = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    try:

        with open(
            image_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ====================================================
        # RUN ACTUAL NEUROVISION PIPELINE
        # ====================================================

        result = analyze_image(
            image_path
        )

        server_url = (
            str(request.base_url)
            .rstrip("/")
        )

        preview_url = (
            f"{server_url}"
            f"/uploads/"
            f"{unique_name}"
        )

        image_width = result["image_width"]
        image_height = result["image_height"]

        # ====================================================
        # DETECTIONS
        # ====================================================

        detections = []

        for detection in result.get(
            "detections",
            []
        ):

            box = detection["bounding_box"]

            detections.append({

                "id":
                    f"det-{detection['id']}",

                "class":
                    detection["yolo_class"],

                "confidence":
                    round(
                        detection["yolo_confidence"] * 100,
                        2
                    ),

                "box": {

                    "x":
                        box["x1"] / image_width,

                    "y":
                        box["y1"] / image_height,

                    "width":
                        (
                            box["x2"] -
                            box["x1"]
                        ) / image_width,

                    "height":
                        (
                            box["y2"] -
                            box["y1"]
                        ) / image_height
                }
            })

        # ====================================================
        # PREDICTIONS
        # ====================================================

        if result.get("detections"):

            first = result["detections"][0]

            yolo_prediction = {
                "class":
                    first["yolo_class"],
                "confidence":
                    round(
                        first["yolo_confidence"] * 100,
                        2
                    )
            }

            resnet_prediction = {
                "class":
                    first["resnet_prediction"],
                "confidence":
                    round(
                        first["resnet_confidence"] * 100,
                        2
                    )
            }

            model_agreement = (
                first["model_agreement"]
            )

            characteristics = (
                first.get(
                    "tumor_characteristics"
                )
            )

        else:

            yolo_prediction = {
                "class": "No Tumor",
                "confidence": 100
            }

            resnet_prediction = {
                "class": "No Tumor",
                "confidence": 100
            }

            model_agreement = True
            characteristics = None

        # ====================================================
        # TUMOR CHARACTERISTICS
        # ====================================================

        if characteristics:

            first_box = (
                result["detections"][0]
                ["bounding_box"]
            )

            x_center = (
                (
                    first_box["x1"] +
                    first_box["x2"]
                ) / 2
                / image_width
            )

            y_center = (
                (
                    first_box["y1"] +
                    first_box["y2"]
                ) / 2
                / image_height
            )

            tumor_characteristics = {

                "width":
                    characteristics["width"],

                "height":
                    characteristics["height"],

                "area":
                    (
                        characteristics["width"] *
                        characteristics["height"]
                    ),

                "relativeArea":
                    round(
                        characteristics[
                            "relative_area"
                        ],
                        2
                    ),

                "areaCategory":
                    characteristics[
                        "area_category"
                    ],

                "center": {
                    "x": round(x_center, 3),
                    "y": round(y_center, 3)
                },

                "horizontal":
                    characteristics[
                        "location"
                    ].split("-")[1],

                "vertical":
                    characteristics[
                        "location"
                    ].split("-")[0],

                "combinedLocation":
                    characteristics[
                        "location"
                    ]
            }

        else:

            tumor_characteristics = {

                "width": 0,
                "height": 0,
                "area": 0,
                "relativeArea": 0,
                "areaCategory": "N/A",

                "center": {
                    "x": 0,
                    "y": 0
                },

                "horizontal": "N/A",
                "vertical": "N/A",
                "combinedLocation": "N/A"
            }

        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        if characteristics:

            risk_score = (
                characteristics["risk_score"]
            )

            location = (
                characteristics["location"]
            )

            location_type = (
                "Central"
                if location == "Middle-Center"
                else "Peripheral"
            )

            risk_analysis = {

                "areaCategory":
                    characteristics[
                        "area_category"
                    ],

                "locationType":
                    location_type,

                "riskLevel":
                    characteristics["risk"],

                "riskScore":
                    risk_score,

                "factors": [

                    (
                        f"{characteristics['area_category']} "
                        f"relative tumor area "
                        f"({characteristics['relative_area']:.2f}%)"
                    ),

                    (
                        f"{location_type} "
                        f"location ({location})"
                    )
                ],

                "disclaimer":
                    (
                        "This research-defined "
                        "stratification is based on "
                        "dataset-derived image "
                        "characteristics and is not a "
                        "clinical diagnosis, prognosis, "
                        "or medical risk assessment."
                    )
            }

        else:

            risk_analysis = {

                "areaCategory": "N/A",
                "locationType": "N/A",
                "riskLevel": "N/A",
                "riskScore": 0,
                "factors": [],

                "disclaimer":
                    (
                        "No tumor region was detected. "
                        "Research risk stratification "
                        "was not performed."
                    )
            }

        # ====================================================
        # GRAD-CAM
        # ====================================================

        if result.get("detections"):

            gradcam = {

                "predictedClass":
                    resnet_prediction["class"],

                "confidence":
                    resnet_prediction["confidence"],

                "available":
                    True
            }

        else:

            gradcam = {

                "predictedClass":
                    "No Tumor",

                "confidence":
                    100,

                "available":
                    False
            }

        # ====================================================
        # FINAL CLASS
        # ====================================================

        if model_agreement:

            final_class = (
                yolo_prediction["class"]
            )

        else:

            final_class = (
                resnet_prediction["class"]
            )

        # ====================================================
        # RESULT IMAGE
        # ====================================================

        result_image_url = None

        result_image_path = result.get(
            "result_image"
        )

        if result_image_path:

            result_filename = os.path.basename(
                result_image_path
            )

            result_image_url = (
                f"{server_url}"
                f"/results/"
                f"{result_filename}"
            )

        # ====================================================
        # ANALYSIS ID
        # ====================================================

        analysis_id = uuid.uuid4().hex

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = {

            "id":
                analysis_id,

            "analysisId":
                analysis_id,

            "image": {

                "name":
                    file.filename,

                "width":
                    image_width,

                "height":
                    image_height,

                "previewUrl":
                    preview_url
            },

            "detections":
                detections,

            "predictions": {

                "yolo":
                    yolo_prediction,

                "resnet":
                    resnet_prediction
            },

            "modelComparison": {

                "agree":
                    model_agreement,

                "yolo":
                    yolo_prediction,

                "resnet":
                    resnet_prediction
            },

            "tumorCharacteristics":
                tumor_characteristics,

            "riskAnalysis":
                risk_analysis,

            "gradcam": {

                **gradcam,

                "imageUrl":
                    result_image_url
            },

            "finalClass":
                final_class,

            "timestamp":
                datetime.now().isoformat()
        }

        # ====================================================
        # SAVE ANALYSIS
        # ====================================================

        ANALYSIS_STORE[
            analysis_id
        ] = response

        return response

    except HTTPException:
        raise

    except Exception as e:

        print(
            "Analysis error:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        await file.close()


# ============================================================
# ASSISTANT REQUEST
# ============================================================

class AssistantRequest(BaseModel):

    question: str | None = None
    context: dict | None = None

    message: str | None = None
    analysis: dict | None = None


# ============================================================
# AI ASSISTANT
# ============================================================

@app.post("/api/assistant")
async def assistant(
    request: AssistantRequest
):

    # ========================================================
    # GET QUESTION
    # ========================================================

    question = (
        request.question
        or request.message
        or ""
    ).strip()

    # ========================================================
    # GET CONTEXT
    # ========================================================

    context = (
        request.context
        or request.analysis
        or {}
    )

    question_lower = question.lower()

    print(
        "\nASSISTANT QUESTION:",
        question
    )

    # ========================================================
    # EXTRACT CURRENT ANALYSIS
    # ========================================================

    final_class = context.get(
        "finalClass",
        "N/A"
    )

    predictions = context.get(
        "predictions",
        {}
    )

    yolo = predictions.get(
        "yolo",
        {}
    )

    resnet = predictions.get(
        "resnet",
        {}
    )

    comparison = context.get(
        "modelComparison",
        {}
    )

    characteristics = context.get(
        "tumorCharacteristics",
        {}
    )

    risk = context.get(
        "riskAnalysis",
        {}
    )

    gradcam = context.get(
        "gradcam",
        {}
    )

    detections = context.get(
        "detections",
        []
    )

    # ========================================================
    # 1. WHAT DID IT DETECT?
    # ========================================================

    if (
        "detect" in question_lower
        or "detected" in question_lower
        or "found" in question_lower
        or "tumor" in question_lower
    ) and (
        "what" in question_lower
        or "which" in question_lower
        or "class" in question_lower
    ):

        if detections:

            answer = (
                f"The model detected a "
                f"{final_class} tumor region. "
                f"YOLO26 predicted "
                f"{yolo.get('class', 'N/A')} "
                f"with {yolo.get('confidence', 0):.2f}% "
                f"confidence, while ResNet50 predicted "
                f"{resnet.get('class', 'N/A')} "
                f"with {resnet.get('confidence', 0):.2f}% "
                f"confidence."
            )

        else:

            answer = (
                "No tumor region was detected "
                "in this MRI by the current "
                "NeuroVision pipeline."
            )

    # ========================================================
    # 2. CONFIDENCE
    # ========================================================

    elif (
        "confidence" in question_lower
        or "certain" in question_lower
        or "sure" in question_lower
    ):

        answer = (
            f"YOLO26 confidence is "
            f"{yolo.get('confidence', 0):.2f}%, "
            f"and ResNet50 confidence is "
            f"{resnet.get('confidence', 0):.2f}%. "
            f"These are model confidence scores, "
            f"not clinical certainty."
        )

    # ========================================================
    # 3. YOLO VS RESNET
    # ========================================================

    elif (
        (
            "yolo" in question_lower
            and "resnet" in question_lower
        )
        or "model comparison" in question_lower
        or "models agree" in question_lower
        or "models disagree" in question_lower
        or "difference between the models"
            in question_lower
    ):

        if comparison.get("agree", False):

            answer = (
                f"Yes, the models agree. "
                f"Both YOLO26 and ResNet50 classify "
                f"the detected region as "
                f"{yolo.get('class', 'N/A')}. "
                f"YOLO26 confidence: "
                f"{yolo.get('confidence', 0):.2f}%. "
                f"ResNet50 confidence: "
                f"{resnet.get('confidence', 0):.2f}%."
            )

        else:

            answer = (
                f"The models disagree. "
                f"YOLO26 predicts "
                f"{yolo.get('class', 'N/A')} "
                f"at {yolo.get('confidence', 0):.2f}%, "
                f"while ResNet50 predicts "
                f"{resnet.get('class', 'N/A')} "
                f"at {resnet.get('confidence', 0):.2f}%."
            )

    # ========================================================
    # 4. SIZE
    # ========================================================

    elif (
        "size" in question_lower
        or "area" in question_lower
        or "how big" in question_lower
    ):

        answer = (
            f"The detected region has an image-space "
            f"area of approximately "
            f"{characteristics.get('area', 0)} pixels "
            f"and occupies about "
            f"{characteristics.get('relativeArea', 0):.2f}% "
            f"of the image. "
            f"The research-defined area category is "
            f"{characteristics.get('areaCategory', 'N/A')}."
        )

    # ========================================================
    # 5. LOCATION
    # ========================================================

    elif (
        "where" in question_lower
        or "location" in question_lower
        or "located" in question_lower
        or "position" in question_lower
    ):

        answer = (
            f"The detected region is positioned at "
            f"{characteristics.get('combinedLocation', 'N/A')}. "
            f"Horizontal position: "
            f"{characteristics.get('horizontal', 'N/A')}. "
            f"Vertical position: "
            f"{characteristics.get('vertical', 'N/A')}."
        )

    # ========================================================
    # 6. RISK
    # ========================================================

    elif (
        "risk" in question_lower
        or "severity" in question_lower
        or "danger" in question_lower
    ):

        answer = (
            f"The research-defined risk category is "
            f"{risk.get('riskLevel', 'N/A')} "
            f"with a score of "
            f"{risk.get('riskScore', 0)}. "
            f"The main factors are: "
            f"{', '.join(risk.get('factors', []))}. "
            f"This is an experimental research "
            f"stratification and is not a clinical "
            f"risk assessment."
        )

    # ========================================================
    # 7. GRAD-CAM
    # ========================================================

    elif (
        "grad-cam" in question_lower
        or "gradcam" in question_lower
        or "heatmap" in question_lower
        or "explainability" in question_lower
        or "explain the prediction" in question_lower
    ):

        if gradcam.get("available", False):

            answer = (
                f"Grad-CAM is available for this analysis. "
                f"It visualizes image regions that contributed "
                f"to the ResNet50 prediction of "
                f"{gradcam.get('predictedClass', 'N/A')}. "
                f"It is an explainability visualization and "
                f"should not be interpreted as clinical proof "
                f"of tumor localization."
            )

        else:

            answer = (
                "Grad-CAM is not available because "
                "no tumor region was detected."
            )

    # ========================================================
    # 8. FINAL RESULT
    # ========================================================

    elif (
        "result" in question_lower
        or "prediction" in question_lower
        or "diagnosis" in question_lower
    ):

        answer = (
            f"The current NeuroVision result is "
            f"{final_class}. "
            f"YOLO26 predicts "
            f"{yolo.get('class', 'N/A')} "
            f"({yolo.get('confidence', 0):.2f}%), "
            f"while ResNet50 predicts "
            f"{resnet.get('class', 'N/A')} "
            f"({resnet.get('confidence', 0):.2f}%)."
        )

    # ========================================================
    # 9. HOW DOES NEUROVISION WORK?
    # ========================================================

    elif (
        "how does" in question_lower
        or "how do you" in question_lower
        or "how does neurovision" in question_lower
        or "pipeline" in question_lower
    ):

        answer = (
            "NeuroVision first processes the uploaded MRI, "
            "then YOLO26 performs tumor localization. "
            "The detected region is analyzed using ResNet50 "
            "for classification. The system then calculates "
            "image-based tumor characteristics, performs the "
            "research-defined risk stratification, and uses "
            "Grad-CAM for visual explanation."
        )

    # ========================================================
    # 10. GENERAL FALLBACK
    # ========================================================

    else:

        answer = (
            f"For this analysis, the current final result "
            f"is {final_class}. "
            f"You can ask me specifically about the detected "
            f"tumor, confidence scores, YOLO26 vs ResNet50, "
            f"tumor size, location, risk analysis, or Grad-CAM."
        )

    print(
        "ASSISTANT ANSWER:",
        answer
    )

    return {
        "answer": answer
    }


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_report(
    analysis_id: str
):

    # ========================================================
    # FIND ANALYSIS
    # ========================================================

    analysis = ANALYSIS_STORE.get(
        analysis_id
    )

    if not analysis:

        raise HTTPException(
            status_code=404,
            detail=(
                "Analysis not found. "
                "Please analyze the MRI again "
                "before generating the report."
            )
        )

    # ========================================================
    # REPORT PATH
    # ========================================================

    report_filename = (
        f"NeuroVision_Report_"
        f"{analysis_id}.pdf"
    )

    report_path = os.path.join(
        REPORTS_DIR,
        report_filename
    )

    # ========================================================
    # DATA
    # ========================================================

    image_data = analysis.get(
        "image",
        {}
    )

    predictions = analysis.get(
        "predictions",
        {}
    )

    yolo = predictions.get(
        "yolo",
        {}
    )

    resnet = predictions.get(
        "resnet",
        {}
    )

    comparison = analysis.get(
        "modelComparison",
        {}
    )

    characteristics = analysis.get(
        "tumorCharacteristics",
        {}
    )

    risk = analysis.get(
        "riskAnalysis",
        {}
    )

    detections = analysis.get(
        "detections",
        []
    )

    # ========================================================
    # PDF
    # ========================================================

    document = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "NeuroVision",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Brain MRI Analysis Report",
            subtitle_style
        )
    )

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "Image Information",
            heading_style
        )
    )

    table = Table(
        [
            [
                "Image",
                str(
                    image_data.get(
                        "name",
                        "N/A"
                    )
                )
            ],
            [
                "Dimensions",
                (
                    f"{image_data.get('width', 'N/A')} × "
                    f"{image_data.get('height', 'N/A')}"
                )
            ],
            [
                "Analysis time",
                str(
                    analysis.get(
                        "timestamp",
                        "N/A"
                    )
                )
            ]
        ],
        colWidths=[160, 340]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(table)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    story.append(
        Paragraph(
            "Analysis Result",
            heading_style
        )
    )

    table = Table(
        [
            [
                "Final classification",
                str(
                    analysis.get(
                        "finalClass",
                        "N/A"
                    )
                )
            ],
            [
                "Tumor detected",
                "Yes" if detections else "No"
            ],
            [
                "YOLO26",
                (
                    f"{yolo.get('class', 'N/A')} "
                    f"({yolo.get('confidence', 0):.2f}%)"
                )
            ],
            [
                "ResNet50",
                (
                    f"{resnet.get('class', 'N/A')} "
                    f"({resnet.get('confidence', 0):.2f}%)"
                )
            ],
            [
                "Models agree",
                "Yes"
                if comparison.get("agree", False)
                else "No"
            ]
        ],
        colWidths=[160, 340]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(table)

    # ========================================================
    # TUMOR CHARACTERISTICS
    # ========================================================

    story.append(
        Paragraph(
            "Tumor Characteristics",
            heading_style
        )
    )

    table = Table(
        [
            [
                "Width",
                str(
                    characteristics.get(
                        "width",
                        "N/A"
                    )
                )
            ],
            [
                "Height",
                str(
                    characteristics.get(
                        "height",
                        "N/A"
                    )
                )
            ],
            [
                "Area",
                str(
                    characteristics.get(
                        "area",
                        "N/A"
                    )
                )
            ],
            [
                "Relative area",
                (
                    f"{characteristics.get('relativeArea', 0):.2f}%"
                )
            ],
            [
                "Area category",
                str(
                    characteristics.get(
                        "areaCategory",
                        "N/A"
                    )
                )
            ],
            [
                "Location",
                str(
                    characteristics.get(
                        "combinedLocation",
                        "N/A"
                    )
                )
            ]
        ],
        colWidths=[160, 340]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(table)

    # ========================================================
    # RISK
    # ========================================================

    story.append(
        Paragraph(
            "Research Risk Analysis",
            heading_style
        )
    )

    table = Table(
        [
            [
                "Risk level",
                str(
                    risk.get(
                        "riskLevel",
                        "N/A"
                    )
                )
            ],
            [
                "Risk score",
                str(
                    risk.get(
                        "riskScore",
                        0
                    )
                )
            ],
            [
                "Location type",
                str(
                    risk.get(
                        "locationType",
                        "N/A"
                    )
                )
            ],
            [
                "Area category",
                str(
                    risk.get(
                        "areaCategory",
                        "N/A"
                    )
                )
            ]
        ],
        colWidths=[160, 340]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(table)

    # ========================================================
    # DETECTIONS
    # ========================================================

    story.append(
        Paragraph(
            "Detected Regions",
            heading_style
        )
    )

    if detections:

        rows = [
            [
                "Region",
                "Class",
                "Confidence"
            ]
        ]

        for detection in detections:

            rows.append(
                [
                    detection.get(
                        "id",
                        "N/A"
                    ),
                    detection.get(
                        "class",
                        "N/A"
                    ),
                    (
                        f"{detection.get('confidence', 0):.2f}%"
                    )
                ]
            )

        table = Table(
            rows,
            colWidths=[120, 200, 180]
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                )
            ])
        )

        story.append(table)

    else:

        story.append(
            Paragraph(
                "No tumor regions were detected.",
                body_style
            )
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "<b>Important:</b> NeuroVision is an academic "
            "research prototype. Its classifications, "
            "image-based tumor characteristics and "
            "research-defined risk stratification are "
            "not a clinical diagnosis, prognosis, "
            "treatment recommendation or medical risk "
            "assessment.",
            body_style
        )
    )

    # ========================================================
    # BUILD
    # ========================================================

    document.build(
        story
    )

    return report_path


# ============================================================
# REPORT - POST
# ============================================================

@app.post(
    "/api/report/{analysis_id}"
)
async def generate_report_post(
    analysis_id: str
):

    report_path = create_report(
        analysis_id
    )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename="NeuroVision_Analysis_Report.pdf"
    )


# ============================================================
# REPORT - GET
# ============================================================

@app.get(
    "/api/report/{analysis_id}"
)
async def generate_report_get(
    analysis_id: str
):

    report_path = create_report(
        analysis_id
    )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename="NeuroVision_Analysis_Report.pdf"
    )