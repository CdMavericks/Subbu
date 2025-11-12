import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import requests
import base64
import time
from deepface import DeepFace
from deepface.modules import verification as dst
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import threading
import json

# --- Configuration ---
VIDEO_SOURCE = "http://192.168.1.37:8080/video"  # Change if needed
AUTH_API_URL = "http://127.0.0.1:8000/api/identify"
MODEL_NAME = "ArcFace"

HANDOFF_THRESHOLD = 0.9
REACQUIRE_THRESHOLD = 0.9
REVERIFY_THRESHOLD = 0.9
REACQUIRE_COOLDOWN = 150
REVERIFY_COOLDOWN = 300

# --- Shared State ---
lock = threading.Lock()
known_tags = {}
cooldowns = {}
student_to_find = {}
last_frame = None  # global frame for MJPEG stream

# --- FastAPI app ---
cam2_app = FastAPI(title="Camera 2 Tracker API")

class HandoffPayload(BaseModel):
    usn: str
    live_embedding: list

@cam2_app.post("/api/initiate_tag")
async def initiate_tag(payload: HandoffPayload):
    global student_to_find
    with lock:
        student_to_find = {"usn": payload.usn, "embedding": payload.live_embedding}
    print(f"[API_CAM2_INFO] Received hand-off command for {payload.usn}")
    return {"status": "handoff_initiated"}

# --- MJPEG Stream ---
def mjpeg_generator():
    global last_frame
    while True:
        if last_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
        time.sleep(0.03)

@cam2_app.get("/stream")
async def stream_video():
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# --- Start API in background ---
def start_api_server():
    print("[INFO] Starting Camera 2 API server on port 8001...")
    uvicorn.run(cam2_app, host="0.0.0.0", port=8001, log_level="warning")

# --- Core Tracker ---
def run_tracker():
    global student_to_find, known_tags, cooldowns, last_frame, face_database

    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_CENTER, text_scale=0.5, text_thickness=1
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {VIDEO_SOURCE}")
        print("Retrying every 5 seconds...")
        while not cap.isOpened():
            time.sleep(5)
            cap = cv2.VideoCapture(VIDEO_SOURCE)

    print("[INFO] Tracker running. Access at http://127.0.0.1:8001/stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, reconnecting...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue

        results = model(frame, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)

        labels = []

        # update cooldowns
        with lock:
            for t_id in list(cooldowns.keys()):
                cooldowns[t_id] -= 1
                if cooldowns[t_id] <= 0:
                    cooldowns.pop(t_id)

        for i, tracker_id in enumerate(tracked_detections.tracker_id):
            bbox = tracked_detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox)
            current_label = "Unknown"
            skip = False

            # --- HANDOFF PHASE ---
            with lock:
                if student_to_find and tracker_id not in known_tags and tracker_id not in cooldowns:
                    usn = student_to_find["usn"]
                    emb = student_to_find["embedding"]
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        try:
                            live = DeepFace.represent(face_crop, model_name=MODEL_NAME,
                                                      enforce_detection=False, detector_backend='skip')
                            dist = dst.find_cosine_distance(np.array(live[0]["embedding"], np.float32),
                                                            np.array(emb, np.float32))
                            if dist < HANDOFF_THRESHOLD:
                                print(f"[HANDOFF] {usn} matched with tracker {tracker_id}")
                                known_tags[tracker_id] = usn
                                student_to_find = {}
                                cooldowns[tracker_id] = REVERIFY_COOLDOWN
                                current_label = usn
                                skip = True
                        except Exception as e:
                            print(f"[HANDOFF_ERR] {e}")

            if skip:
                labels.append(current_label)
                continue

            # --- RE-VERIFY / RE-ACQUIRE ---
            with lock:
                if tracker_id in known_tags:
                    current_label = known_tags[tracker_id]
                    if tracker_id not in cooldowns:
                        cooldowns[tracker_id] = REVERIFY_COOLDOWN
                else:
                    # attempt reacquire
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        try:
                            _, buf = cv2.imencode(".jpg", face_crop)
                            b64 = base64.b64encode(buf).decode("utf-8")
                            res = requests.post(AUTH_API_URL, json={"image_base64": b64}, timeout=0.5)
                            if res.status_code == 200:
                                data = res.json()
                                if data.get("distance", 1.0) < REACQUIRE_THRESHOLD:
                                    usn = data["usn"]
                                    print(f"[RE-ACQUIRE] Found {usn}")
                                    known_tags[tracker_id] = usn
                                    cooldowns[tracker_id] = REVERIFY_COOLDOWN
                                    current_label = usn
                        except Exception:
                            cooldowns[tracker_id] = REACQUIRE_COOLDOWN

            labels.append(current_label)

        annotated = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated = label_annotator.annotate(scene=annotated, detections=tracked_detections, labels=labels)

        _, buffer = cv2.imencode(".jpg", annotated)
        last_frame = buffer.tobytes()

# --- Main Run ---
if __name__ == "__main__":
    face_database = {}
    try:
        with open("face_database.json", "r") as f:
            face_database = json.load(f)
        print(f"[INFO_CAM2] Loaded DB with {len(face_database)} entries.")
    except Exception as e:
        print(f"[WARN] No valid face_database.json found: {e}")

    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    time.sleep(2)
    run_tracker()
