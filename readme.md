# Real-Time Object Tracking — YOLOv8 + ByteTrack

Multi-object tracking system using YOLOv8 for detection and ByteTrack
for stable ID assignment across video frames. Deployed via FastAPI REST
API and containerized with Docker.

## Results
| Metric      | Score  |
|-------------|--------|
| MOTA        | 100%   |
| IDF1        | 100%   |
| ID Switches | 0      |
| People tracked | 7   |

## Tech Stack
Python · YOLOv8 · OpenCV · ByteTrack · FastAPI · Docker

## Project Structure

## Quick Start
```bash
git clone https://github.com/manish123456-ctrl/Object-Tracker
cd Object-Tracker
pip install -r requirements.txt
uvicorn src.api:app --port 8000
```

## API Usage
POST /track — upload a video, receive tracked output

## Features
- YOLOv8 real-time object detection
- ByteTrack stable ID assignment across frames
- Colour trajectory trails per tracked object
- FastAPI REST endpoint for video upload
- Dockerized for deployment

![Demo](assets/demo.png)