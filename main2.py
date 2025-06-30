import cv2
import numpy as np
import mediapipe as mp
import geocoder
from datetime import datetime, time, timedelta

# Initialize data storage
gender_detection_data = {'Male': [], 'Female': []}

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

# Function to get live location (based on IP address)
def get_live_location():
    g = geocoder.ip('me')
    if g.latlng:
        latitude, longitude = g.latlng
        return latitude, longitude
    else:
        return None, None

# Function to log alerts with location data
def log_alert_with_location(alert_message, male_count=0, female_count=0):
    latitude, longitude = get_live_location()
    location_info = f"Location: Lat {latitude}, Long {longitude}" if latitude and longitude else "Location: Unknown"
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{current_time} - {alert_message} - Males: {male_count}, Females: {female_count} - {location_info}\n")

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

# Function to check if it's currently night-time between 7:30 PM and 6:00 AM
def is_night_time():
    current_time = datetime.now().time()
    night_start = time(19, 30)  # 7:30 PM
    night_end = time(6, 0)    # 6:00 AM
    return night_start <= current_time or current_time <= night_end

# Function to detect if a person is in a hotspot area
def detect_hotspot_area(x, y, w, h, frame_width, frame_height):
    # Define the hotspot area as the lower half of the frame
    hotspot_area_y_start = int(frame_height * 0.5)  # Adjust this as needed
    hotspot_area_y_end = frame_height  # Bottom part of the frame

    # Check if the face is in the hotspot area
    if y + h > hotspot_area_y_start and y + h < hotspot_area_y_end:
        return True
    return False

# Function to process frames and perform gender detection
def process_frame(frame, alert_flags, last_snapshot_time, snapshot_interval=10, frame_number=0, video_writer=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False
    detected_gender = None
    frame_height, frame_width = frame.shape[:2]  # Get frame dimensions

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_confidence = gender_preds[0].max()
        gender = gender_list[gender_preds[0].argmax()]

        confidence_threshold = 0.6
        if gender_confidence > confidence_threshold:
            detected_gender = gender
            if gender == 'Male':
                male_count += 1
                color = (255, 0, 0)  # Blue for male
            else:
                female_count += 1
                color = (0, 255, 0)  # Green for female

            label = f"{gender} ({gender_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # Check if the person is in the hotspot area
            if detect_hotspot_area(x, y, w, h, frame_width, frame_height):
                log_alert_with_location("Person in Hotspot Area", male_count, female_count)

    # Process hand gestures using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            sos_detected = detect_sos_gesture_and_position(landmarks, frame_height)
            if sos_detected:
                alert_flags["sos_detected"] = True
                break

    lone_woman_detected = len(faces) == 1 and female_count == 1

    # Display alerts if conditions are met
    alerts = []  # List to store the alerts

    if alert_flags["sos_detected"]:
        alerts.append("SOS Situation Detected!")
        log_alert_with_location("SOS Situation Detected!", male_count, female_count)
        alert_flags["sos_detected"] = False  # Reset flag after logging to avoid duplicate entries

        # Start recording if SOS is detected and video_writer is not yet initialized
        if video_writer is None:
            # Define video codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('sos_detected_video.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    if lone_woman_detected and is_night_time():
        alerts.append("Lone Woman Detected at Night")
        log_alert_with_location("Lone Woman Detected at Night", male_count, female_count)

    if male_count >= 3 and female_count == 1:
        alerts.append("Alert! More Male to Female Ratio Found (3:1 or More)")
        log_alert_with_location("Alert! More Male to Female Ratio Found (3:1 or More)", male_count, female_count)

    # Display the alerts within the alert box in the right corner with transparent black background
    alert_box_width = 300
    alert_box_height = 200  # Increased height for better visibility
    alert_box_x = frame.shape[1] - alert_box_width - 10 
    alert_box_y = frame.shape[0] - alert_box_height - 10 
    overlay = frame.copy() 
    cv2.rectangle(overlay, (alert_box_x, alert_box_y), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), -1) 
    alpha = 0.6  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display alerts in the alert box
    y_offset = alert_box_y + 20  # Start with some padding from the top
    max_chars_per_line = 30  # Maximum characters per line
    for alert in alerts:
        lines = []
        while len(alert) > max_chars_per_line:
            idx = alert.rfind(' ', 0, max_chars_per_line)
            if idx == -1:
                idx = max_chars_per_line
            lines.append(alert[:idx])
            alert = alert[idx+1:]
        lines.append(alert)
        for line in lines:
            cv2.putText(frame, line, (alert_box_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

    # Take a snapshot if required at intervals
    if lone_woman_detected and (datetime.now() - last_snapshot_time).seconds >= snapshot_interval:
        snapshot_filename = f"lone_woman_snapshot_{frame_number}.jpg"
        cv2.imwrite(snapshot_filename, frame)
        last_snapshot_time = datetime.now()

    # Write frame to video file if recording
    if video_writer is not None:
        video_writer.write(frame)

    return frame, last_snapshot_time, video_writer

# Main function to start the detection process
def start_detection():
    cap = cv2.VideoCapture(0)
    frame_number = 0
    alert_flags = {"sos_detected": False}
    last_snapshot_time = datetime.now()
    video_writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame, last_snapshot_time, video_writer = process_frame(frame, alert_flags, last_snapshot_time, frame_number=frame_number, video_writer=video_writer)

        # Display the frame
        cv2.imshow('Gender Detection with Alerts', frame)
        frame_number += 1

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()