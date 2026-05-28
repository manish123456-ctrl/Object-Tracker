from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
import supervision as sv
import cv2, uuid, os
from collections import defaultdict

app   = FastAPI(title="Object Tracker API", version="1.0")
model = YOLO("yolov8n.pt")

os.makedirs("data/input",  exist_ok=True)
os.makedirs("data/output", exist_ok=True)

COLORS = [
    (0,255,150),(255,100,0),(0,100,255),
    (255,0,180),(0,220,255),(180,255,0)
]

def run_tracking(input_path: str, output_path: str):
    tracker         = sv.ByteTrack()
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    history         = defaultdict(list)

    cap = cv2.VideoCapture(input_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    total_ids = set()

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
                x1,y1,x2,y2 = map(int, box)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                history[tid].append((cx, cy))
                if len(history[tid]) > 40:
                    history[tid].pop(0)
                color = COLORS[tid % len(COLORS)]
                pts   = history[tid]
                for j in range(1, len(pts)):
                    cv2.line(frame, pts[j-1], pts[j], color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

            labels = [f"#{t}" for t in detections.tracker_id]
            frame  = box_annotator.annotate(frame, detections)
            frame  = label_annotator.annotate(frame, detections, labels)

        out.write(frame)

    cap.release()
    out.release()
    return len(total_ids)

@app.get("/")
def health():
    return {"status": "ok", "model": "yolov8n"}

@app.post("/track")
async def track_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "Only mp4/avi/mov supported")

    uid          = str(uuid.uuid4())[:8]
    input_path   = f"data/input/{uid}_{file.filename}"
    output_path  = f"data/output/{uid}_tracked.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    total_ids = run_tracking(input_path, output_path)
    os.remove(input_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="tracked_output.mp4",
        headers={"X-Total-IDs": str(total_ids)}
    )