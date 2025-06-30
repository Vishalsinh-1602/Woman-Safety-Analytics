import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, time
from twilio.rest import Client

# Twilio configuration
TWILIO_ACCOUNT_SID = 'AC72921747fdb3b4f3ff6c7282c65a1cbe'
TWILIO_AUTH_TOKEN = 'c90c6ce524cb8e1ebb1014f8eae88837'
TWILIO_PHONE_NUMBER = '+19708409136'
RECIPIENT_PHONE_NUMBERS = ['+91 9313379176']  # Add your phone numbers here

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to send SMS alert
def send_sms_alert(message):
    for number in RECIPIENT_PHONE_NUMBERS:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=number
        )

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

# Function to log alerts to a single text file
def log_alert(alert_message):
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{current_time} - {alert_message}\n")

# Function to detect SOS gesture and check hand position relative to shoulder
def detect_sos_gesture_and_position(landmarks, frame_height):
    if len(landmarks) == 21:
        hand_y = landmarks[9][1] * frame_height  # Wrist position as a reference
        shoulder_y = 0.3 * frame_height  # Assume shoulder is around 30% of frame height (adjust as needed)
        
        if hand_y < shoulder_y:
            if (landmarks[8][2] < landmarks[6][2] and
                landmarks[12][2] < landmarks[10][2] and
                landmarks[16][2] < landmarks[14][2]):
                return True
    return False

# Function to check if it's currently night-time between 12 AM and 5 AM
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(0, 0)  # 12:00 AM
    night_end = time(5, 0)    # 5:00 AM
    return night_start <= current_time <= night_end

# Function to process frames and perform gender detection
def process_frame(frame, alert_flags):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False

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
                color = (255, 0, 0)  # Blue for male
            else:
                female_count += 1
                color = (0, 255, 0)  # Green for female

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Gender Distribution - Male: {male_count} Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Process hand gestures using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            frame_height = frame.shape[0]  # Get the frame height
            sos_detected = detect_sos_gesture_and_position(landmarks, frame_height)
            if sos_detected:
                alert_flags["sos_detected"] = True
                break

    lone_woman_detected = len(faces) == 1 and female_count == 1

    # Define alert messages and positions
    alerts = []
    if alert_flags["sos_detected"]:
        alerts.append(("SOS Situation Detected!", (10, frame.shape[0] - 150), 1.0, (0, 0, 255)))
        log_alert("SOS Situation Detected!")
        send_sms_alert("SOS Situation Detected!")
        alert_flags["sos_detected"] = False  # Reset flag after logging to avoid duplicate entries

    if lone_woman_detected and is_night_time():
        alerts.append(("Lone Woman Detected at Night", (10, frame.shape[0] - 120), 1.0, (0, 0, 255)))
        log_alert("Lone Woman Detected at Night")

    if female_count == 1 and male_count >= 2:
        alerts.append(("Alert! 1 Woman Surrounded by 2 or More Men", (10, frame.shape[0] - 90), 1.0, (0, 0, 255)))
        log_alert("Alert! 1 Woman Surrounded by 2 or More Men")

    if male_count >= 3 and female_count == 1:
        alerts.append(("Alert! More Male to Female Ratio Found (3:1 or More)", (10, frame.shape[0] - 60), 1.0, (0, 0, 255)))
        log_alert("Alert! More Male to Female Ratio Found (3:1 or More)")

    # Display alerts
    for message, position, font_scale, color in alerts:
        cv2.putText(frame, message, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

    return frame

# Function to handle video capture from webcam or video file
def process_video(source):
    video = cv2.VideoCapture(source)
    screen_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cv2.namedWindow("Gender Detection with Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gender Detection with Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize alert flags
    alert_flags = {
        "sos_detected": False
    }

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame or end of video reached")
            break

        frame = process_frame(frame, alert_flags)
        frame = cv2.resize(frame, (int(screen_width), int(screen_height)))
        cv2.imshow("Gender Detection with Gesture Recognition", frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Process video file first (replace with the actual path to your video file)
#process_video('D:\\Gender_Detection\\test.mp4')

# Then switch to live webcam feed (using 0 as the source for the webcam)
process_video(0)
