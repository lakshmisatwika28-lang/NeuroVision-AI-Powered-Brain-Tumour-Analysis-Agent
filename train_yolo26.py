from ultralytics import YOLO

# Load pretrained YOLO26 nano model
model = YOLO("yolo26n.pt")

# Sanity training run
results = model.train(
    data="dataset/NeuroVision_YOLO_v2/data.yaml",
    epochs=3,
    imgsz=640,
    batch=8,
    project="runs",
    name="neurovision_yolo26_sanity",
    pretrained=True,
    workers=2
)

print("\nYOLO26 SANITY TRAINING COMPLETE!")
print("Check the results inside:")
print("runs/neurovision_yolo26_sanity/")