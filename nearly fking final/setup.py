import os

print("Starting ClassSight setup for offline enrollment...")

# --- Configuration ---
MAIN_IMAGE_FOLDER = "CSE-C" 
NUM_STUDENTS = 320 # How many student folders to create (e.g., up to CS_320)

# Add any other general directories your project might need
OTHER_DIRECTORIES = [] 
# Example: OTHER_DIRECTORIES = ["output_data"]

# --- Create Main Image Folder ---
if not os.path.exists(MAIN_IMAGE_FOLDER):
    try:
        os.makedirs(MAIN_IMAGE_FOLDER)
        print(f"  + Created main image directory: '{MAIN_IMAGE_FOLDER}'")
    except OSError as e:
        print(f"  [ERROR] Failed to create directory '{MAIN_IMAGE_FOLDER}': {e}")
        # If we can't create the main folder, don't try to create sub-folders
        MAIN_IMAGE_FOLDER = None 
else:
    print(f"  - Main image directory '{MAIN_IMAGE_FOLDER}' already exists.")

# --- Create Student Sub-Folders ---
if MAIN_IMAGE_FOLDER: # Only proceed if the main folder exists or was created
    print(f"\nCreating student sub-folders inside '{MAIN_IMAGE_FOLDER}' (CS_001 to CS_{NUM_STUDENTS:03d})...")
    created_count = 0
    skipped_count = 0
    for i in range(1, NUM_STUDENTS + 1):
        # Format the student ID with leading zeros (e.g., CS_001, CS_010, CS_100)
        student_id = f"CS_{i:03d}" 
        student_folder_path = os.path.join(MAIN_IMAGE_FOLDER, student_id)
        
        if not os.path.exists(student_folder_path):
            try:
                os.makedirs(student_folder_path)
                created_count += 1
            except OSError as e:
                print(f"  [ERROR] Failed to create directory '{student_folder_path}': {e}")
        else:
            skipped_count += 1
            
    print(f"  + Created {created_count} new student sub-folders.")
    if skipped_count > 0:
        print(f"  - Skipped {skipped_count} existing student sub-folders.")
    print(f"    IMPORTANT: You must now manually add student photos into their respective sub-folders (e.g., '{MAIN_IMAGE_FOLDER}/CS_001/photo.jpg').")


# --- Create Other Directories ---
print("\nChecking other directories...")
for directory in OTHER_DIRECTORIES:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"  + Created directory: '{directory}'")
        except OSError as e:
            print(f"  [ERROR] Failed to create directory '{directory}': {e}")
    else:
        print(f"  - Directory '{directory}' already exists. Skipping.")


print("\nSetup complete for necessary directories.")
print("Remember to manually add student photos into their sub-folders before running enroll.py.")

