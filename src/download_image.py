from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

result = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=False,
    conf=0.25
)

img = result[0].plot()
cv2.imwrite("data/output/detected.jpg", img)

boxes = result[0].boxes
print(f"Detected {len(boxes)} objects")
for box in boxes:
    label = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    print(f"  {label}: {conf:.2f}")