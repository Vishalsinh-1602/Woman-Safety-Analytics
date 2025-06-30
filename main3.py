import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, time

# Load pre-trained model files for gender detection
gender_proto = "D:/Gender_Detection/gender_deploy.prototxt"
gender_model = "D:/Gender_Detection/gender_net.caffemodel"

# Load the gender model using OpenCV's dnn module
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Define model mean values for the gender model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Define gender labels
gender_list = ['Male', 'Female']

# Load OpenCV's pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Initialize SOS detection state variables
sos_start_time = None
sos_active = False
sos_mistake_start_time = None

# Function to log alerts to a single text file
def log_alert(alert_message):
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{current_time} - {alert_message}\n")

# Function to detect SOS gesture (open palm above face box for 2+ seconds)
def detect_sos_gesture(landmarks, face_box_y):
    global sos_start_time, sos_active

    if len(landmarks) == 21:
        # Get y-coordinates of the wrist and fingertips
        wrist_y = landmarks[0][1]  # Normalized value
        index_finger_tip_y = landmarks[8][1]
        middle_finger_tip_y = landmarks[12][1]
        ring_finger_tip_y = landmarks[16][1]
        pinky_finger_tip_y = landmarks[20][1]

        # Check if the hand is above the upper line of the face box and fingers are extended (open palm)
        if wrist_y < face_box_y and all([
            index_finger_tip_y < wrist_y,
            middle_finger_tip_y < wrist_y,
            ring_finger_tip_y < wrist_y,
            pinky_finger_tip_y < wrist_y
        ]):
            # Start timer if not already started
            if sos_start_time is None:
                sos_start_time = datetime.now()
            elif (datetime.now() - sos_start_time).seconds >= 2:
                sos_active = True  # SOS gesture confirmed
                return True
        else:
            sos_start_time = None  # Reset if the gesture is not maintained
    return False

# Function to detect mistake in SOS gesture (crossed hands on chest)
def detect_sos_mistake(landmarks):
    global sos_mistake_start_time

    if len(landmarks) == 21:
        # Get y-coordinates of the wrist and fingertips
        wrist_y = landmarks[0][1]  # Normalized value
        index_finger_tip_y = landmarks[8][1]
        middle_finger_tip_y = landmarks[12][1]
        ring_finger_tip_y = landmarks[16][1]
        pinky_finger_tip_y = landmarks[20][1]

        # Check for crossed hands on chest
        if wrist_y > index_finger_tip_y and wrist_y > middle_finger_tip_y:
            if sos_mistake_start_time is None:
                sos_mistake_start_time = datetime.now()
            elif (datetime.now() - sos_mistake_start_time).seconds >= 2:
                return True
        else:
            sos_mistake_start_time = None  # Reset if the gesture is not maintained
    return False

# Function to check if it's currently night-time between 12 AM and 5 AM
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(0, 0)  # 12:00 AM
    night_end = time(5, 0)    # 5:00 AM
    return night_start <= current_time <= night_end

# Function to process frames and perform gender detection
def process_frame(frame, alert_flags):
    global sos_active
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False
    sos_mistake_detected = False

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_confidence = gender_preds[0].max()
        gender = gender_list[gender_preds[0].argmax()]

        confidence_threshold = 0.6
        if gender_confidence > confidence_threshold:
            if gender == 'Male':
                male_count += 1
                color = (255, 0, 0)  # Blue
            else:
                female_count += 1
                color = (0, 255, 0)  # Green

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Process hand gestures using MediaPipe and detect SOS gesture
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Adjust SOS gesture detection to use the face box's upper y-coordinate
                if detect_sos_gesture(landmarks, y / frame.shape[0]):  # Normalize face box y-coordinate
                    sos_detected = True
                    alert_flags["sos_detected"] = True

                # Detect SOS mistake
                if detect_sos_mistake(landmarks):
                    sos_mistake_detected = True
                    alert_flags["sos_mistake_detected"] = True

    cv2.putText(frame, f"Gender Distribution - Male: {male_count} Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    lone_woman_detected = len(faces) == 1 and female_count == 1

    # Collect all alerts
    alerts = []

    if alert_flags["sos_detected"]:
        alerts.append("SOS Situation Detected!")
        log_alert("SOS Situation Detected!")
        alert_flags["sos_detected"] = False  # Reset flag after logging to avoid duplicate entries

    if alert_flags["sos_mistake_detected"]:
        alerts.append("SOS Gesture Mistake!")
        log_alert("SOS Gesture Mistake!")
        alert_flags["sos_mistake_detected"] = False  # Reset flag after logging

    if lone_woman_detected and is_night_time():
        alerts.append("Lone Woman Detected at Night")
        log_alert("Lone Woman Detected at Night")

    if female_count == 1 and male_count >= 2:
        alerts.append("Alert! 1 Woman Surrounded by 2 or More Men")
        log_alert("Alert! 1 Woman Surrounded by 2 or More Men")

    if male_count >= 3 and female_count == 1:
        alerts.append("Alert! More Male to Female Ratio Found (3:1 or More)")
        log_alert("Alert! More Male to Female Ratio Found (3:1 or More)")

    # Draw alerts on the right side of the display
    if alerts:
        box_width = 250
        box_height = 200  # Increased height for better visibility
        box_x = frame.shape[1] - box_width - 10
        box_y = frame.shape[0] - box_height - 10
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)

        max_alerts_to_display = 4
        displayed_alerts = alerts[-max_alerts_to_display:]  # Show up to 4 alerts at a time
        line_height = 30
        for i, alert in enumerate(displayed_alerts):
            cv2.putText(frame, alert, (box_x + 10, box_y + 30 + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1, cv2.LINE_AA)


# Open camera for capturing video
cap = cv2.VideoCapture(0)
alert_flags = {"sos_detected": False, "sos_mistake_detected": False}  # Initialize a dictionary to hold alert flags

# Create a named window and set it to full screen
cv2.namedWindow("Gender Detection with Alerts", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gender Detection with Alerts", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    process_frame(frame, alert_flags)

    cv2.imshow("Gender Detection with Alerts", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
