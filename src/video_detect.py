from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/input/people.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video: {width}x{height} at {fps}fps")

out = cv2.VideoWriter(
    "data/output/detected_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    out.write(frame)
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Done! Total frames: {frame_count}")
print("Saved to data/output/detected_video.mp4")