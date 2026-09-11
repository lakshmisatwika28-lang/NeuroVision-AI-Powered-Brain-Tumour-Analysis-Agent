from ultralytics import YOLO

model = YOLO("models/yolo26_best.pt")

results = model.val(
    data="dataset/NeuroVision_YOLO_v2/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    workers=2
)

print("\nYOLO26 TEST EVALUATION COMPLETE!")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")