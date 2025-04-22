import streamlit as st
import cv2
import time
import numpy as np
import mediapipe as mp
from streamlit.components.v1 import html

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    'models/deploy_gender.prototxt',
    'models/gender_net.caffemodel'
)
gender_classes = ['Male', 'Female']

# JavaScript for beep sound
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

# Configuration sidebar
st.sidebar.header("Configuration")
ear_threshold = st.sidebar.slider("EAR Threshold", 0.0, 1.0, 0.18)
mar_threshold = st.sidebar.slider("MAR Threshold", 0.0, 2.0, 0.75)
eyes_closed_threshold = st.sidebar.slider("Eyes Closed Threshold (sec)", 1, 10, 2)
yawn_duration_threshold = st.sidebar.slider("Yawn Threshold (sec)", 1, 10, 3)
distraction_threshold = st.sidebar.slider("Distraction Threshold (sec)", 1, 10, 2)

# MediaPipe landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

def calculate_ear(eye_points, landmarks):
    # Calculate eye aspect ratio
    horizontal = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
    vertical1 = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
    vertical2 = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def calculate_mar(lip_points, landmarks):
    # Calculate mouth aspect ratio
    horizontal = np.linalg.norm(landmarks[lip_points[0]] - landmarks[lip_points[6]])
    vertical1 = np.linalg.norm(landmarks[lip_points[2]] - landmarks[lip_points[9]])
    vertical2 = np.linalg.norm(landmarks[lip_points[4]] - landmarks[lip_points[7]])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# App state
if 'run' not in st.session_state:
    st.session_state.run = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# UI controls
st.title("Real-time Driver Monitoring System")
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Monitoring") and not st.session_state.run:
        st.session_state.run = True
        st.session_state.cap = cv2.VideoCapture(0)
with col2:
    if st.button("Stop Monitoring") and st.session_state.run:
        st.session_state.run = False
        if st.session_state.cap:
            st.session_state.cap.release()

# Main processing loop
if st.session_state.run and st.session_state.cap:
    cap = st.session_state.cap
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = []
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))
            
            landmarks = np.array(landmarks)
            
            # Eye detection
            left_ear = calculate_ear(LEFT_EYE, landmarks)
            right_ear = calculate_ear(RIGHT_EYE, landmarks)
            
            # Yawn detection
            mar = calculate_mar(LIPS, landmarks)
            
            # Gender detection
            face_roi = frame[50:250, 50:250]
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                       (78.4263377603, 87.7689143744, 114.895847746), 
                                       swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_classes[gender_preds[0].argmax()]
            
            # Visualization and alerts
            cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 2)
            cv2.putText(frame, f'Gender: {gender}', (50, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if (left_ear + right_ear)/2 < ear_threshold:
                cv2.putText(frame, 'Eyes Closed!', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                html(beep_js)
                
            if mar > mar_threshold:
                cv2.putText(frame, 'Yawning!', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                html(beep_js)
                
            # Draw landmarks
            for idx in LEFT_EYE + RIGHT_EYE + LIPS:
                cv2.circle(frame, landmarks[idx], 1, (0, 255, 0), -1)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB")
        st.experimental_rerun()
        
else:
    if st.session_state.cap:
        st.session_state.cap.release()
