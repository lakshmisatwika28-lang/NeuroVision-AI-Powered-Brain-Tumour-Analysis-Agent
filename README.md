# NeuroVision

## Explainable Brain MRI Classification and Tumor Risk Analysis

NeuroVision is an AI-based research prototype for automated analysis of brain MRI images. The system combines tumor detection, deep learning classification, explainable AI, tumor characteristic extraction, and research-defined risk analysis into a single web application.

> **Disclaimer:** NeuroVision is an academic and research prototype. It is not a medical diagnostic system and should not be used for clinical diagnosis, prognosis, treatment decisions, or emergency medical decisions.

---

## Features

- Brain tumor detection and localization using YOLO26
- Tumor classification using a custom CNN
- Tumor classification using ResNet50 transfer learning
- Comparison of predictions from both models
- Grad-CAM based explainability
- Tumor characteristic extraction from bounding boxes
- Research-defined tumor risk stratification
- React and Vite frontend
- FastAPI backend
- AI analysis assistant
- PDF analysis report generation
- Analysis history

---

## System Architecture

```text
Brain MRI Image
       |
       v
Image Quality Check
       |
       v
YOLO26 Detection
       |
       +----------------------+
       |                      |
       v                      v
Tumor Localization       Tumor Class
       |
       v
Tumor Region Extraction
       |
       v
+-------------------------+
|                         |
v                         v
Custom CNN            ResNet50
|                         |
+------------+------------+
             |
             v
      Model Comparison
             |
             v
     Tumor Classification
             |
             v
    Tumor Characteristics
             |
             v
     Research Risk Analysis
             |
             v
          Grad-CAM
             |
             v
       Final Analysis
             |
             v
       Web Application


REPOSITORY STRUCTURE

NeuroVision/
|
├── frontend/
|   ├── src/
|   ├── public/
|   ├── package.json
|   ├── package-lock.json
|   └── vite.config.js
|
├── backend/
|   ├── main.py
|   ├── uploads/
|   └── reports/
|
├── models/
|   ├── yolo26_best.pt
|   ├── custom_cnn.py
|   ├── custom_cnn_best.pth
|   └── resnet50_best.pth
|
├── dataset/
|   └── NeuroVision_YOLO_v2/
|
├── neurovision_pipeline.py
├── train_yolo26.py
├── test_yolo26.py
├── train_custom_cnn.py
├── evaluate_custom_cnn.py
├── train_resnet50.py
├── evaluate_resnet50.py
├── gradcam_resnet50.py
├── evaluate_gradcam.py
├── tumour_characteristics.py
├── risk_analysis.py
├── prepare_classifier_regions.py
├── extract_tumour_regions.py
├── dataset_audit.py
├── create_test_split.py
└── visualize_dataset.py


DATASET

The project uses the following public dataset:

MRI for Brain Tumor with Bounding Boxes

Kaggle:

https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes

The dataset contains brain MRI images with YOLO-format bounding-box annotations.


DATASET SET UP

Download the dataset from Kaggle and extract it.
The original dataset has a structure similar to:

Dataset/
|
├── Train/
|   ├── Glioma/
|   ├── Meningioma/
|   ├── No Tumor/
|   └── Pituitary/
|
└── Val/
    ├── Glioma/
    ├── Meningioma/
    ├── No Tumor/
    └── Pituitary/


The dataset needs to be converted into the YOLO format used by the project.

Run the dataset audit:
        python dataset_audit.py
Create the train, validation, and test split:
        python create_test_split.py
The prepared dataset follows:
dataset/
└── NeuroVision_YOLO_v2/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/



Requirements

Python 3.13 was used during development.
Create a virtual environment:
       python -m venv .venv

Activate it on Windows:
       .venv\Scripts\activate

Install the required Python packages:
       pip install torch torchvision
       pip install ultralytics
       pip install numpy pandas pillow opencv-python
       pip install scikit-learn matplotlib
       pip install grad-cam
       pip install fastapi uvicorn python-multipart
       pip install reportlab

For the frontend:
      cd frontend
      npm install


## YOLO26

YOLO26 is used for tumor detection and localization.

The trained model is stored at:

models/yolo26_best.pt

The dataset configuration is:

dataset/NeuroVision_YOLO_v2/data.yaml

To train the model:

    python train_yolo26.py

The repository already contains the trained checkpoint, so retraining is not required to run the existing pipeline.

To evaluate the model:

    python test_yolo26.py


## Tumor Region Preparation

Tumor regions are extracted using the ground-truth YOLO annotations for classifier training.

Run:

    python prepare_classifier_regions.py

This creates:

    classifier_regions/
    ├── train/
    │   ├── Glioma/
    │   ├── Meningioma/
    │   └── Pituitary/
    ├── val/
    │   ├── Glioma/
    │   ├── Meningioma/
    │   └── Pituitary/
    └── test/
        ├── Glioma/
        ├── Meningioma/
        └── Pituitary/

No Tumor images do not produce tumor crops and are handled at the detection stage.


## Custom CNN

The project includes a custom convolutional neural network.

Train the model:

    python train_custom_cnn.py

Evaluate it:

    python evaluate_custom_cnn.py

The trained model is stored at:

    models/custom_cnn_best.pth


## ResNet50

ResNet50 is used as a transfer-learning classifier.

The final classification layer is modified for the three tumor classes:

    Glioma
    Meningioma
    Pituitary

Train:

    python train_resnet50.py

Evaluate:

    python evaluate_resnet50.py

The trained checkpoint is:

    models/resnet50_best.pth


## Grad-CAM

Grad-CAM is used to visualize image regions contributing to the ResNet50 prediction.

Run:

    python gradcam_resnet50.py "path/to/image.jpg"

Example:

    python gradcam_resnet50.py "classifier_regions/test/Glioma/sample.jpg"

Generated visualizations are saved in:

    gradcam_results/

Grad-CAM is used as an explainability visualization and should not be interpreted as proof of clinical tumor localization.


## Tumor Characteristics

Tumor characteristics are calculated from YOLO bounding-box annotations.

Run:

    python tumour_characteristics.py

The generated file is:

    tumor_characteristics.csv

The analysis includes:

- Bounding-box width
- Bounding-box height
- Bounding-box area
- Relative tumor area
- Tumor center
- Horizontal location
- Vertical location
- Combined location


## Risk Analysis

NeuroVision includes a research-defined risk stratification module based on image-derived characteristics.

Run:

    python risk_analysis.py

The output is:

    tumor_risk_analysis.csv

The module considers:

- Relative tumor area
- Tumor location
- Research-defined risk score

The resulting categories are:

    Lower
    Moderate
    Higher

These categories are intended only for research and demonstration purposes and are not clinical risk assessments.


## Running the Complete Pipeline

The complete analysis pipeline can be executed with:

    python neurovision_pipeline.py "path/to/mri_image.jpg"

Example:

    python neurovision_pipeline.py "pit_test_img.png"

The pipeline performs:

    MRI Image
        |
        v
    YOLO26 Detection
        |
        v
    Tumor Localization
        |
        v
    Tumor Crop
        |
        v
    ResNet50 Classification
        |
        v
    Tumor Characteristics
        |
        v
    Research Risk Analysis
        |
        v
    Grad-CAM
        |
        v
    Final Analysis

Generated analysis images are saved in:

    neurovision_results/


## Running the Backend

From the project root:

    .venv\Scripts\activate

Start the FastAPI server:

    uvicorn backend.main:app --reload --port 8000

The backend runs at:

    http://localhost:8000

Main API endpoints:

    POST /api/analyze
    POST /api/assistant
    POST /api/report/{analysis_id}
    GET  /api/report/{analysis_id}


## Running the Frontend

Open another terminal.

Navigate to the frontend:

    cd frontend

Install dependencies:

    npm install

Start the development server:

    npm run dev

The frontend normally runs at:

    http://localhost:5173

Make sure the FastAPI backend is running before uploading an MRI image for analysis.


## Frontend and Backend Workflow

    React Frontend
          |
          | MRI Upload
          v
    FastAPI Backend
          |
          v
    NeuroVision Pipeline
          |
          +-- YOLO26
          +-- ResNet50
          +-- Tumor Characteristics
          +-- Risk Analysis
          +-- Grad-CAM
          |
          v
    Analysis Response
          |
          v
    Results Dashboard


## Model Performance

The models were evaluated on the held-out test set.

### Custom CNN

    Accuracy  : 85.31%
    Precision : 86.38%
    Recall    : 85.31%
    F1 Score  : 85.17%

### ResNet50

    Accuracy  : 95.83%
    Precision : 95.83%
    Recall    : 95.83%
    F1 Score  : 95.83%

### YOLO26

    mAP@50    : 92.03%
    mAP@50-95 : 70.95%

These results represent performance on the project's dataset and should not be interpreted as clinical performance.


## Technologies Used

### Machine Learning

- Python
- PyTorch
- Torchvision
- Ultralytics YOLO26
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Pillow

### Explainable AI

- Grad-CAM

### Backend

- FastAPI
- Uvicorn
- ReportLab

### Frontend

- React
- Vite
- JavaScript
- CSS
- Lucide React
- Recharts


## Limitations

- The project has not undergone clinical validation.
- The dataset has limited size and diversity.
- Performance may vary on MRI images from different scanners, institutions, or imaging protocols.
- Model confidence does not represent clinical certainty.
- Bounding boxes do not provide complete tumor segmentation.
- Grad-CAM provides attribution visualization rather than causal explanation.
- The risk categories are research-defined and are not medical risk assessments.


## Future Work

Possible improvements include:

- Training with larger and more diverse MRI datasets
- External validation on independent datasets
- Dedicated tumor segmentation
- Improved confidence calibration
- More robust explainability evaluation
- Uncertainty estimation
- Secure database storage
- Authentication and access control
- Cloud deployment


## License

This project is intended for academic and research purposes.

The dataset is obtained from Kaggle. Refer to the original dataset page for the applicable dataset license and usage conditions.


## Project

NeuroVision is an academic project focused on computer vision, deep learning, explainable AI, medical image analysis, and full-stack AI application development.