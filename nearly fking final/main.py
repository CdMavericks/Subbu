import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
from deepface.modules import verification as dst  # The correct import
import json
import numpy as np
import cv2
import base64
import os
import time
import requests  # <-- NEW IMPORT

# --- Configuration ---
DATABASE_NAME = "face_database.json"
MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 0.99
CAMERA_2_API_URL = "http://127.0.0.1:8001/api/initiate_tag" # <-- CORRECT HOST/PORT
# ---------------------

app = FastAPI(title="Camera 1: Auth Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

face_database = {}

class ImagePayload(BaseModel):
    image_base64: str

@app.on_event("startup")
def load_database():
    global face_database
    if not os.path.exists(DATABASE_NAME):
        print(f"[ERROR] Database file not found: {DATABASE_NAME}")
        print("Please run enroll.py first.")
    else:
        with open(DATABASE_NAME, 'r') as f:
            face_database = json.load(f)
        print(f"[INFO] Successfully loaded database with {len(face_database)} students.")
    
    try:
        print("[INFO] Pre-loading AI model...")
        DeepFace.build_model(MODEL_NAME)
        print("[INFO] AI Model pre-loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not pre-load model: {e}")

def decode_base64_to_image(base64_string):
    try:
        if "," in base64_string:
            _, encoded = base64_string.split(",", 1)
        else:
            encoded = base64_string
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Invalid base64 string: {e}")
        return None

@app.post("/api/identify")
async def identify_face(payload: ImagePayload):
    start_time = time.time()
    
    live_image = decode_base64_to_image(payload.image_base64)
    if live_image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    try:
        live_embedding_obj = DeepFace.represent(
            img_path=live_image,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        live_embedding = live_embedding_obj[0]["embedding"]
        
    except ValueError:
        raise HTTPException(status_code=404, detail="No face detected in image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    best_match_usn = None
    best_match_distance = float('inf')

    for usn, saved_embeddings_list in face_database.items():
        for saved_embedding in saved_embeddings_list:
            try:
                distance = dst.find_cosine_distance(
                    np.array(live_embedding, dtype=np.float32), 
                    np.array(saved_embedding, dtype=np.float32)
                )
                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_usn = usn
            except Exception as e:
                print(f"[ERROR] Error during distance calc: {e}")
                continue

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000

    print(f"[INFO] Best match: {best_match_usn} with distance {best_match_distance:.4f} (Time: {processing_time:.0f}ms)")

    if best_match_distance < DISTANCE_THRESHOLD:
        student_name = f"Student {best_match_usn}" # Placeholder
        
        # --- START OF NEW HAND-OFF LOGIC ---
        print(f"[HAND-OFF] Telling Camera 2 to tag {best_match_usn}...")
        try:
            handoff_payload = {
                "usn": best_match_usn,
                # We must convert the numpy list to a standard Python list
                "live_embedding": live_embedding 
            }
            # We send the request but don't wait for it
            # This makes our API respond super fast
            requests.post(CAMERA_2_API_URL, json=handoff_payload, timeout=0.5)
            
        except requests.exceptions.RequestException as e:
            # This is not a critical error, so we just log it
            print(f"[WARNING] Could not connect to Camera 2 for hand-off: {e}")
        # --- END OF NEW HAND-OFF LOGIC ---
        
        return {
            "status": "success",
            "usn": best_match_usn,
            "name": student_name, 
            "distance": float(best_match_distance) # Cast to float
        }
    else:
        print(f"[INFO] Match failed threshold. Best distance {best_match_distance:.4f} > {DISTANCE_THRESHOLD}")
        raise HTTPException(status_code=404, detail="Face not recognized. Please enroll.")

if __name__ == "__main__":
    # Runs Camera 1 Server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
