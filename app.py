from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import face_recognition
import os
import numpy as np
import sqlite3
import time
from datetime import datetime

app = Flask(__name__)

# --- Settings ---
UPLOAD_FOLDER = 'known_faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Database Setup ---
def init_db():
    try:
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                face_count INTEGER,
                alert_status TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

init_db()

# --- Global Variables ---
known_face_encodings = []
known_face_names = []

# --- Add Face Function ---
def add_face_to_system(filepath, filename, init=False):
    global known_face_encodings, known_face_names
    try:
        img = cv2.imread(filepath)
        if img is None:
            if not init: os.remove(filepath)
            return

        # Resize large images
        height, width = img.shape[:2]
        if width > 1000:
            ratio = 1000 / width
            img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            if not init: print(f"✅ NEW FACE ADDED: {name}")
        else:
            if not init: os.remove(filepath)

    except Exception as e:
        print(f"❌ Error: {e}")

# --- Load Faces ---
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                add_face_to_system(filepath, filename, init=True)

load_known_faces()

camera = cv2.VideoCapture(0)

# Global Vars
last_face_locations = []
last_face_names = []
last_alert_status = False
frame_counter = 0

def generate_frames():
    global frame_counter, last_face_locations, last_face_names, last_alert_status
    last_save_time = time.time()
    
    while True:
        try:
            success, frame = camera.read()
            if not success or frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- AI Logic ---
            frame_counter += 1
            if frame_counter % 2 == 0: 
                face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=1)
                last_face_locations = face_locations
                
                # BUG FIX: Watchlist එක හිස් නම් Alert එක අනිවාර්යයෙන්ම False කරන්න ඕන
                if known_face_encodings:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    current_names = []
                    alert = False
                    
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                        name = "Unknown"
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]
                            alert = True
                        current_names.append(name)
                    
                    last_face_names = current_names
                    last_alert_status = alert
                else:
                    # මෙන්න මේ පේළිය තමයි කලින් අඩු වෙලා තිබ්බේ!
                    last_face_names = ["Visitor"] * len(face_locations)
                    last_alert_status = False  # <--- FORCE STOP ALERT

            # --- Drawing ---
            for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                if last_alert_status and name != "Unknown" and name != "Visitor":
                    color = (0, 0, 255) 
                    text = f"ALERT: {name}"
                else:
                    color = (0, 255, 0)
                    text = "Visitor"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Count Display
            real_count = len(last_face_locations)
            cv2.rectangle(frame, (10, 10), (280, 60), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, f'Faces: {real_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if last_alert_status:
                cv2.putText(frame, "!!! RED ALERT !!!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Database Save
            if time.time() - last_save_time > 5:
                if real_count > 0:
                    try:
                        status = "HIGH ALERT" if last_alert_status else "Normal"
                        with sqlite3.connect('face_database.db') as conn:
                            cursor = conn.cursor()
                            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            cursor.execute('INSERT INTO logs (timestamp, face_count, alert_status) VALUES (?, ?, ?)', (now, real_count, status))
                            conn.commit()
                    except:
                        pass
                last_save_time = time.time()

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                       
        except Exception as e:
            print(f"Stream Error: {e}")

# --- Routes ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/get_logs')
def get_logs():
    try:
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        return jsonify(rows)
    except:
        return jsonify([])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return redirect('/')
    file = request.files['file']
    if file.filename == '': return redirect('/')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        add_face_to_system(filepath, file.filename)
        return redirect('/')

@app.route('/reset_watchlist')
def reset_watchlist():
    global known_face_encodings, known_face_names, last_alert_status
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(file_path)
        
        # මෙමරි එක සුද්ද කරනවා වගේම ALERT එකත් False කරනවා
        known_face_encodings = []
        known_face_names = []
        last_alert_status = False # <--- වැදගත්ම දේ
        
        print("System Reset Done.")
    except:
        pass
    return redirect('/')

@app.route('/clear_data')
def clear_data():
    try:
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM logs")
        conn.commit()
        conn.close()
    except:
        pass
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)