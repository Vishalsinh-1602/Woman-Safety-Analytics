import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, time, timedelta
import geopy
import geocoder  # Used to get the current location
from geopy.geocoders import Nominatim

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

# Function to get live location using geopy and geocoder
def get_live_location():
    try:
        # Get latitude and longitude from IP-based location using geocoder
        location = geocoder.ip('me').latlng
        if location:
            latitude, longitude = location
            geolocator = Nominatim(user_agent="geoapiExercises")  # Initialize geopy Nominatim
            place_info = geolocator.reverse((latitude, longitude), language='en')  # Get place details
            
            if place_info and place_info.raw.get('address'):
                address = place_info.raw['address']
                
                # Extract components like place name, district, state, pin code, country
                city_town = address.get('village') or address.get('town') or address.get('city', 'Unknown City/Town')
                district = address.get('county', 'Unknown District')
                state = address.get('state', 'Unknown State')
                pin_code = address.get('postcode', 'Unknown Pin Code')
                country = address.get('country', 'Unknown Country')
                
                # Create a formatted location string with all details
                location_details = f"{city_town}, {district}, {state}, {country}, Pin Code: {pin_code}"
            else:
                location_details = "Location name unavailable"
            
            return latitude, longitude, city_town, district, state, country, pin_code  # Return all details
        else:
            return None, None, "Location unavailable", "", "", "", ""
    except Exception as e:
        return None, None, str(e), "", "", "", ""

# Function to log alerts to a single text file
def log_alert(alert_message, male_count=0, female_count=0):
    latitude, longitude, city_town, district, state, country, pin_code = get_live_location()  # Get detailed location
    with open("alerts_log.txt", "a") as log_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(
            f"{current_time} - {alert_message} - Males: {male_count}, Females: {female_count} "
            f"- Location: {latitude}, {longitude} - City/Town: {city_town}, District: {district}, "
            f"State: {state}, Country: {country}, Pin Code: {pin_code}\n"
        )

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

# Function to process frames and perform gender detection
def process_frame(frame, alert_flags, last_snapshot_time, snapshot_interval=10, frame_number=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    male_count = 0
    female_count = 0
    lone_woman_detected = False
    sos_detected = False
    detected_gender = None

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

    # Display alerts if conditions are met
    alerts = []  # List to store the alerts

    if alert_flags["sos_detected"]:
        alerts.append("SOS Situation Detected!")
        log_alert("SOS Situation Detected!", male_count, female_count)
        alert_flags["sos_detected"] = False  # Reset flag after logging to avoid duplicate entries

    if lone_woman_detected and is_night_time():
        alerts.append("Lone Woman Detected at Night")
        log_alert("Lone Woman Detected at Night", male_count, female_count)

    if male_count >= 3 and female_count == 1:
        alerts.append("Alert! More Male to Female Ratio Found (3:1 or More)")
        log_alert("Alert! More Male to Female Ratio Found (3:1 or More)", male_count, female_count)

    # Display the alerts within the alert box in the right corner with transparent black background
    alert_box_width = 300
    alert_box_height = 200  # Increased height for better visibility
    alert_box_x = frame.shape[1] - alert_box_width - 10 
    alert_box_y = frame.shape[0] - alert_box_height - 10 
    overlay = frame.copy() 
    cv2.rectangle(overlay, (alert_box_x, alert_box_y), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), -1)
    alpha = 0.6  # Transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y0 = alert_box_y + 30
    for alert in alerts:
        cv2.putText(frame, alert, (alert_box_x + 10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y0 += 30

    # Return frame without video writer
    return frame, alert_flags, last_snapshot_time

# Main function to capture video and process frames
def main():
    cap = cv2.VideoCapture(0)
    alert_flags = {"sos_detected": False}
    last_snapshot_time = datetime.now() - timedelta(seconds=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, alert_flags, last_snapshot_time = process_frame(frame, alert_flags, last_snapshot_time)

        cv2.imshow('Gender Detection & SOS Gesture Recognition', frame)

        # Exit loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()