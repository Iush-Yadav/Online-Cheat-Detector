import cv2
import mediapipe as mp
import time
import platform

def beep():
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, 500)
    elif platform.system() == 'Darwin':
        import os
        os.system('afplay /System/Library/Sounds/Glass.aiff')
    else:
        print('\a')

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark constants
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Thresholds and parameters
MAX_GLANCES = 5
GLANCE_COOLDOWN = 1.0
SUSTAINED_LOOK_DURATION = 4
CALIBRATION_DURATION = 3.0  # seconds

# State
cap = cv2.VideoCapture(0)
last_eye_direction = "CENTER"
last_count_time = time.time()
sustained_start_time = None
left_count = 0
right_count = 0
warning_active = False
calibration_data = []

calibrated = False
neutral_left = 0.5
neutral_right = 0.5
calibration_start_time = time.time()

def get_relative_iris_pos(iris_x, eye_left_x, eye_right_x):
    eye_width = eye_right_x - eye_left_x
    return (iris_x - eye_left_x) / eye_width

def determine_eye_direction(relative_pos, neutral_pos):
    if relative_pos < neutral_pos - 0.12:
        return "LEFT"
    elif relative_pos > neutral_pos + 0.12:
        return "RIGHT"
    return "CENTER"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    current_time = time.time()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Positions
        left_iris_x = landmarks[LEFT_IRIS].x
        right_iris_x = landmarks[RIGHT_IRIS].x
        left_eye_left_x = landmarks[LEFT_EYE_LEFT].x
        left_eye_right_x = landmarks[LEFT_EYE_RIGHT].x
        right_eye_left_x = landmarks[RIGHT_EYE_LEFT].x
        right_eye_right_x = landmarks[RIGHT_EYE_RIGHT].x

        # Relative positions
        rel_left = get_relative_iris_pos(left_iris_x, left_eye_left_x, left_eye_right_x)
        rel_right = get_relative_iris_pos(right_iris_x, right_eye_left_x, right_eye_right_x)

        if not calibrated:
            # Calibration Phase
            calibration_data.append((rel_left, rel_right))
            elapsed = current_time - calibration_start_time
            cv2.putText(frame, f"Calibrating... Look Straight ({int(CALIBRATION_DURATION - elapsed)}s left)",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if elapsed >= CALIBRATION_DURATION:
                avg_left = sum(x[0] for x in calibration_data) / len(calibration_data)
                avg_right = sum(x[1] for x in calibration_data) / len(calibration_data)
                neutral_left = avg_left
                neutral_right = avg_right
                calibrated = True
            cv2.imshow("Eye Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Direction detection
        left_eye_direction = determine_eye_direction(rel_left, neutral_left)
        right_eye_direction = determine_eye_direction(rel_right, neutral_right)

        eye_direction = "CENTER"
        if left_eye_direction == "LEFT" or right_eye_direction == "LEFT":
            eye_direction = "LEFT"
        elif left_eye_direction == "RIGHT" or right_eye_direction == "RIGHT":
            eye_direction = "RIGHT"

        # Sustained look detection
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

        # Glance counting
        if eye_direction != last_eye_direction and eye_direction in ("LEFT", "RIGHT") and (current_time - last_count_time > GLANCE_COOLDOWN):
            if eye_direction == "LEFT":
                left_count = min(left_count + 1, MAX_GLANCES)
            elif eye_direction == "RIGHT":
                right_count = min(right_count + 1, MAX_GLANCES)
            last_count_time = current_time

        last_eye_direction = eye_direction

        # Warnings
        warning_active = (
            left_count >= MAX_GLANCES or
            right_count >= MAX_GLANCES or
            sustained_duration >= SUSTAINED_LOOK_DURATION
        )

        # Display
        y = 50
        cv2.putText(frame, f"Eye Dir: {eye_direction}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Left glances: {left_count}/{MAX_GLANCES}", (30, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Right glances: {right_count}/{MAX_GLANCES}", (30, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if sustained_start_time:
            cv2.putText(frame, f"Sustained: {int(sustained_duration)}s/{SUSTAINED_LOOK_DURATION}s",
                        (30, y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        if warning_active:
            cv2.putText(frame, "WARNING: SUSPICIOUS EYE MOVEMENT DETECTED", 
                        (30, y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            beep()
    else:
        cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Calibrated Eye Movement Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
