import os
import json
from deepface import DeepFace
import time

print("[INFO] Starting enrollment script...")

# --- Configuration ---
# 1. Point this to your main folder
STUDENT_IMAGES_PATH = r"C:\Users\User\Desktop\HACKLOOP-PHASE1\Mark - 01\enrollment_data" 

# 2. This is the pre-trained model we'll use to create faceprints
# ArcFace is the current industry standard for accuracy.
MODEL_NAME = "ArcFace" 

# 3. This is the output file
DATABASE_NAME = "face_database.json"
# ---------------------

# This will hold all our data in memory before we save it
# Format: { "usn": [ [embedding1], [embedding2], ... ] }
face_database = {}

start_time = time.time()

# os.walk is the perfect tool for nested folders.
# It will go through CSE-C and enter each sub-folder.
# root: The current folder (e.g., '.../CSE-C/CS_124')
# dirs: Any sub-folders inside (we don't need this)
# files: A list of filenames (e.g., ['frontal_id.jpg', 'left_angle.jpg'])
for root, dirs, files in os.walk(STUDENT_IMAGES_PATH):
    
    # We only care about the sub-folders (CS_124, CS_125, etc.)
    # not the top-level folder (CSE-C)
    if root == STUDENT_IMAGES_PATH:
        continue
        
    # Get the USN from the folder name
    # os.path.basename(root) gives 'CS_124'
    usn = os.path.basename(root)
    
    if usn not in face_database:
        face_database[usn] = [] # Initialize an empty list for this student
    
    print(f"\n[INFO] Processing student: {usn}")

    # Now, loop over all photos *for this student*
    for file in files:
        # Just a check to ignore non-image files like .DS_Store
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(root, file)
        
        try:
            # This is the core AI part!
            # It loads the image, finds the face, and converts it
            # into a 512-dimension vector (the faceprint).
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=MODEL_NAME,
                enforce_detection=True # Fails if no face is found
            )
            
            # The output of DeepFace.represent is a list containing a dict.
            # We just want the 'embedding' vector itself.
            faceprint = embedding[0]["embedding"]
            
            # Add the faceprint to our student's list
            face_database[usn].append(faceprint)
            print(f"  > Successfully processed {file} (1 face added)")

        except ValueError as e:
            # This error triggers if enforce_detection=True and no face is found
            print(f"  [WARNING] No face detected in {image_path}. Skipping.")
        except Exception as e:
            print(f"  [ERROR] An error occurred with {image_path}: {e}")

# --- Save to Disk ---
try:
    with open(DATABASE_NAME, 'w') as f:
        json.dump(face_database, f, indent=4)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*30)
    print(f"[SUCCESS] Enrollment Complete!")
    print(f"Total students processed: {len(face_database)}")
    print(f"Data saved to: {DATABASE_NAME}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print("="*30)

except Exception as e:
    print(f"\n[ERROR] Could not save database file: {e}")