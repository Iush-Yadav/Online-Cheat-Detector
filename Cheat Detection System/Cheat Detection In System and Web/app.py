import cv2
import mediapipe as mp
import time
import platform
import threading
from flask import Flask, render_template, jsonify, Response

app = Flask(__name__)

# Beep sound
def beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, 200)
    else:
        print('\a')

def continuous_beep(duration_sec=3):
    end_time = time.time() + duration_sec
    while time.time() < end_time:
        beep()
        time.sleep(0.3)

# Logging
LOG_FILE = "eye_movement_log.txt"
event_log = []

def log_event(text):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {text}"
    event_log.append(entry)
    if len(event_log) > 100:
        event_log.pop(0)
    with open(LOG_FILE, "a") as f:
        f.write(entry + "\n")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Landmarks
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Params
GLANCE_LIMIT = 5
SUSTAINED_DURATION_THRESHOLD = 4
CALIBRATION_DURATION = 3.0

# State
neutral_left = 0.5
neutral_right = 0.5
calibration_data = []
calibration_start_time = time.time()

status = {
    "eye_direction": "CENTER",
    "left_glances": 0,
    "right_glances": 0,
    "sustained_duration": 0,
    "warning_active": False,
    "calibrated": False,
    "calibration_progress": 0,
    "events": [],
}

def get_relative_iris_pos(iris_x, eye_left_x, eye_right_x):
    eye_width = eye_right_x - eye_left_x
    return (iris_x - eye_left_x) / eye_width if eye_width else 0.5

def determine_eye_direction(relative_pos, neutral):
    if relative_pos < neutral - 0.12:
        return "LEFT"
    elif relative_pos > neutral + 0.12:
        return "RIGHT"
    return "CENTER"

def generate_frames():
    global calibration_start_time, neutral_left, neutral_right, calibration_data
    last_eye_direction = "CENTER"
    last_count_time = time.time()
    sustained_start_time = None
    left_count = 0
    right_count = 0
    warning_active = False
    calibrated = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            current_time = time.time()

            sustained_duration = 0
            eye_direction = "CENTER"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_iris_x = landmarks[LEFT_IRIS].x
                right_iris_x = landmarks[RIGHT_IRIS].x
                left_eye_left_x = landmarks[LEFT_EYE_LEFT].x
                left_eye_right_x = landmarks[LEFT_EYE_RIGHT].x
                right_eye_left_x = landmarks[RIGHT_EYE_LEFT].x
                right_eye_right_x = landmarks[RIGHT_EYE_RIGHT].x

                rel_left = get_relative_iris_pos(left_iris_x, left_eye_left_x, left_eye_right_x)
                rel_right = get_relative_iris_pos(right_iris_x, right_eye_left_x, right_eye_right_x)

                if not calibrated:
                    calibration_data.append((rel_left, rel_right))
                    elapsed = current_time - calibration_start_time
                    status["calibration_progress"] = min(elapsed / CALIBRATION_DURATION, 1.0)
                    cv2.putText(frame, f"Calibrating... {int(CALIBRATION_DURATION - elapsed)}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    if elapsed >= CALIBRATION_DURATION:
                        neutral_left = sum(x[0] for x in calibration_data) / len(calibration_data)
                        neutral_right = sum(x[1] for x in calibration_data) / len(calibration_data)
                        calibrated = True
                        status["calibrated"] = True
                        log_event("Calibration complete.")
                else:
                    left_eye_dir = determine_eye_direction(rel_left, neutral_left)
                    right_eye_dir = determine_eye_direction(rel_right, neutral_right)

                    if left_eye_dir == "LEFT" or right_eye_dir == "LEFT":
                        eye_direction = "LEFT"
                    elif left_eye_dir == "RIGHT" or right_eye_dir == "RIGHT":
                        eye_direction = "RIGHT"

                    # Sustained detection
                    if eye_direction in ("LEFT", "RIGHT"):
                        if eye_direction == last_eye_direction:
                            if sustained_start_time is None:
                                sustained_start_time = current_time
                            sustained_duration = current_time - sustained_start_time
                        else:
                            sustained_start_time = current_time
                            sustained_duration = 0
                    else:
                        sustained_start_time = None
                        sustained_duration = 0

                    # Glance detection
                    if eye_direction != last_eye_direction and eye_direction in ("LEFT", "RIGHT") and (current_time - last_count_time > 1):
                        if eye_direction == "LEFT":
                            left_count += 1
                            log_event(f"Left glance #{left_count}")
                        elif eye_direction == "RIGHT":
                            right_count += 1
                            log_event(f"Right glance #{right_count}")
                        last_count_time = current_time

                    # Trigger warning for 5 glances
                    if (left_count >= GLANCE_LIMIT or right_count >= GLANCE_LIMIT) and not warning_active:
                        warning_active = True
                        direction = "LEFT" if left_count >= GLANCE_LIMIT else "RIGHT"
                        log_event(f"WARNING: {direction} glance threshold exceeded!")
                        threading.Thread(target=continuous_beep, args=(3,), daemon=True).start()
                        left_count = 0
                        right_count = 0
                        threading.Thread(target=lambda: (time.sleep(3), setattr(status, 'warning_active', False)), daemon=True).start()

                    last_eye_direction = eye_direction

                    if sustained_duration >= SUSTAINED_DURATION_THRESHOLD:
                        log_event(f"WARNING: Sustained {eye_direction} gaze for {int(sustained_duration)}s")
                        sustained_start_time = None
                        sustained_duration = 0

                    # Draw overlays
                    y = 40
                    cv2.putText(frame, f"Dir: {eye_direction}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"L: {left_count}/5  R: {right_count}/5", (30, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if sustained_start_time:
                        cv2.putText(frame, f"Sustained: {int(sustained_duration)}s", (30, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

                    if warning_active:
                        cv2.putText(frame, "WARNING!", (30, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)

                    status.update({
                        "eye_direction": eye_direction,
                        "left_glances": left_count,
                        "right_glances": right_count,
                        "sustained_duration": int(sustained_duration),
                        "warning_active": warning_active,
                        "events": event_log[-10:]
                    })
            else:
                cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
