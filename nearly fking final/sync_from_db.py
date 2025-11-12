import mysql.connector
import requests
import os
from pathlib import Path

# ---------------- CONFIG ----------------
OUTPUT_DIR = "enrollment_images"

DB_HOST = "mysqldb-nmam-events.h.aivencloud.com"
DB_USER = "avnadmin"
DB_PASSWORD = "AVNS_YgAtatHm_yR2IzgTItR"
DB_NAME = "hackloop"
DB_PORT = 25203

URL_COLUMNS = [
    'url_frontal',
    'url_left',
    'url_right',
    'url_up',
    'url_down'
]
# -----------------------------------------

def download_images_from_db():
    """Connects to MySQL, fetches enrollment data, and downloads each student’s face images."""
    print(f"Connecting to database: {DB_NAME} at {DB_HOST}:{DB_PORT} ...")

    try:
        # ✅ Connect to database
        db = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            ssl_disabled=False  # must be False for Aiven Cloud
        )

        cursor = db.cursor(dictionary=True)
        print("✅ Database connection successful.")

        # ✅ Fetch data
        columns_to_fetch = ['USN'] + URL_COLUMNS
        query = f"SELECT {', '.join(columns_to_fetch)} FROM enrollments"
        cursor.execute(query)
        enrollments = cursor.fetchall()

        if not enrollments:
            print("⚠ No enrollment data found in the table.")
            return

        # ✅ Setup local folder
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        print(f"\n📁 Saving data inside: {OUTPUT_DIR}")
        print(f"🔽 Found {len(enrollments)} records to process.\n")

        for row in enrollments:
            usn = row['USN']
            branch_code = usn[3:5].upper() if len(usn) >= 5 else "UNK"

            # 📂 Folder hierarchy: enrollment_images/{year}/{branch}/{USN}
            year_folder = Path(OUTPUT_DIR) / f"Year_{usn[2]}"  # e.g., Year_2
            branch_folder = year_folder / branch_code
            student_folder = branch_folder / usn

            student_folder.mkdir(parents=True, exist_ok=True)
            print(f"🧩 Processing {usn} -> {student_folder}")

            for pose_col in URL_COLUMNS:
                url = row.get(pose_col)
                if not url:
                    print(f"   ⚠ Missing URL for {pose_col}")
                    continue

                pose_name = pose_col.split('_')[-1]
                file_path = student_folder / f"{pose_name}.jpg"

                try:
                    response = requests.get(url, timeout=8)
                    response.raise_for_status()

                    with open(file_path, 'wb') as f:
                        f.write(response.content)

                    print(f"   ✅ {pose_name}.jpg downloaded")

                except requests.exceptions.RequestException as e:
                    print(f"   ❌ Failed to download {pose_name}: {e}")

        print("\n🎯 All images downloaded successfully.")

    except mysql.connector.Error as err:
        print(f"\n❌ Database connection error: {err}")

    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()
            print("🔒 Database connection closed.")


if __name__ == "__main__":
    download_images_from_db()
