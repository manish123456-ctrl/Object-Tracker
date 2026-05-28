from ultralytics import YOLO
import supervision as sv
import cv2
from collections import defaultdict

model   = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

COLORS = [
    (0, 255, 150), (255, 100, 0),   (0, 100, 255),
    (255, 0, 180), (0, 220, 255),   (180, 255, 0),
    (255, 180, 0), (100, 0, 255)
]

trajectory_history = defaultdict(list)
MAX_TRAIL = 40

cap    = cv2.VideoCapture("data/input/people.mp4")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "data/output/tracked_trails.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

frame_count = 0
total_ids   = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results    = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.5]
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is not None:
        total_ids.update(detections.tracker_id.tolist())

        for box, tid in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            trajectory_history[tid].append((cx, cy))
            if len(trajectory_history[tid]) > MAX_TRAIL:
                trajectory_history[tid].pop(0)

            color = COLORS[tid % len(COLORS)]
            pts   = trajectory_history[tid]

            for j in range(1, len(pts)):
                alpha = j / len(pts)
                thick = max(1, int(alpha * 3))
                cv2.line(frame, pts[j-1], pts[j], color, thick)

            cv2.circle(frame, (cx, cy), 4, color, -1)

        labels = [f"#{tid}" for tid in detections.tracker_id]
        frame  = box_annotator.annotate(frame, detections)
        frame  = label_annotator.annotate(frame, detections, labels)

    out.write(frame)
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Frame {frame_count} | IDs seen: {len(total_ids)}")

cap.release()
out.release()
print(f"\nDone! Frames: {frame_count}")
print(f"Unique people tracked: {len(total_ids)}")
print("Saved to data/output/tracked_trails.mp4")