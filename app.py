import streamlit as st
import cv2
import dlib
import time
import numpy as np
from streamlit.components.v1 import html

# Define the JavaScript code for playing a beep sound
beep_js = """
<script>
function beep() {
    var context = new AudioContext();
    var oscillator = context.createOscillator();
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(1000, context.currentTime);
    oscillator.connect(context.destination);
    oscillator.start();
    setTimeout(function() { oscillator.stop(); }, 500);
}
beep();
</script>
"""

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks):
    top_lip = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(50, 53)])
    bottom_lip = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(65, 68)])
    mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 54)])

    A = np.linalg.norm(bottom_lip[1] - top_lip[1])
    B = np.linalg.norm(bottom_lip[0] - top_lip[0])
    C = np.linalg.norm(bottom_lip[2] - top_lip[2])
    D = np.linalg.norm(mouth[3] - mouth[0])

    mar = (A + B + C) / (3.0 * D)
    return mar

@st.cache_resource
def load_models(shape_predictor_path, gender_prototxt_path, gender_caffemodel_path):
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(shape_predictor_path)
    gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt_path, gender_caffemodel_path)
    return face_detector, landmark_predictor, gender_net

# Sidebar configuration
st.sidebar.header("Configuration")
shape_predictor_path = st.sidebar.text_input("Shape Predictor Path", "models/dataset.dat")
gender_prototxt_path = st.sidebar.text_input("Gender Prototxt Path", "models/deploy_gender.prototxt")
gender_caffemodel_path = st.sidebar.text_input("Gender Caffemodel Path", "models/gender_net.caffemodel")

ear_threshold = st.sidebar.slider("EAR Threshold", 0.0, 1.0, 0.5)
mar_threshold = st.sidebar.slider("MAR Threshold", 0.0, 2.0, 0.6)
eyes_closed_threshold = st.sidebar.slider("Eyes Closed Threshold (sec)", 1, 10, 2)
yawn_duration_threshold = st.sidebar.slider("Yawn Threshold (sec)", 1, 10, 3)
distraction_threshold = st.sidebar.slider("Distraction Threshold (sec)", 1, 10, 2)

gender_classes = ['Male', 'Female']

# Load models
try:
    face_detector, landmark_predictor, gender_net = load_models(
        shape_predictor_path, gender_prototxt_path, gender_caffemodel_path
    )
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Main app
st.title("Real-time Driver Monitoring System")

# Initialize session state
if 'run' not in st.session_state:
    st.session_state.run = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Monitoring") and not st.session_state.run:
        st.session_state.run = True
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.landmark_unavailable_start_time = None
        st.session_state.eyes_closed_start_time = None
        st.session_state.yawn_start_time = None
with col2:
    if st.button("Stop Monitoring") and st.session_state.run:
        st.session_state.run = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None

# Main processing loop
if st.session_state.run and st.session_state.cap is not None:
    cap = st.session_state.cap
    ret, frame = cap.read()
    
    if ret:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        if len(faces) > 0:
            st.session_state.landmark_unavailable_start_time = None
            for face in faces:
                landmarks = landmark_predictor(gray, face)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_roi = frame[y:y+h, x:x+w]
                
                # Gender detection
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                           (78.4263377603, 87.7689143744, 114.895847746), 
                                           swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_classes[gender_preds[0].argmax()]
                cv2.putText(frame, f'Gender: {gender}', (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw facial landmarks
                for i in range(68):
                    x_point = landmarks.part(i).x
                    y_point = landmarks.part(i).y
                    color = (0, 0, 255) if 48 <= i <= 67 else (0, 255, 0)
                    cv2.circle(frame, (x_point, y_point), 1, color, -1)
                
                # Yawn detection
                mar = calculate_mar(landmarks)
                if mar > mar_threshold:
                    cv2.putText(frame, 'Yawning Detected', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if st.session_state.yawn_start_time is None:
                        st.session_state.yawn_start_time = time.time()
                    else:
                        if time.time() - st.session_state.yawn_start_time >= yawn_duration_threshold:
                            html(beep_js)
                else:
                    st.session_state.yawn_start_time = None
                
                # Eye detection
                left_ear = ((landmarks.part(39).y - landmarks.part(37).y) + 
                           (landmarks.part(40).y - landmarks.part(38).y)) / (2 * (landmarks.part(41).x - landmarks.part(36).x))
                right_ear = ((landmarks.part(45).y - landmarks.part(43).y) + 
                            (landmarks.part(46).y - landmarks.part(44).y)) / (2 * (landmarks.part(47).x - landmarks.part(42).x))
                
                if left_ear < ear_threshold and right_ear < ear_threshold:
                    cv2.putText(frame, 'Eyes Closed', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if st.session_state.eyes_closed_start_time is None:
                        st.session_state.eyes_closed_start_time = time.time()
                    else:
                        if time.time() - st.session_state.eyes_closed_start_time >= eyes_closed_threshold:
                            html(beep_js)
                else:
                    st.session_state.eyes_closed_start_time = None
                    cv2.putText(frame, 'Eyes Open', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if st.session_state.landmark_unavailable_start_time is None:
                st.session_state.landmark_unavailable_start_time = time.time()
            else:
                if time.time() - st.session_state.landmark_unavailable_start_time >= distraction_threshold:
                    cv2.putText(frame, 'You are distracted', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    html(beep_js)
        
        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB")
        st.experimental_rerun()
    else:
        st.error("Failed to capture frame")
        st.session_state.run = False
        st.session_state.cap.release()

# Release resources when app stops
if not st.session_state.run and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None