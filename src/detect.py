from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

frame = cv2.imread("data/input/test_frame.jpg")
print(f"Image size: {frame.shape}")

results = model("data/input/test_frame.jpg")

for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} objects")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]

        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

cv2.imwrite("data/output/detected.jpg", frame)
print("Saved to data/output/detected.jpg")