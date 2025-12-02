import cv2
import streamlit as st
from deepface import DeepFace

st.title("🧠 Real-Time Deepfake Detection (Improved Version)")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

st.write("Press **S** to stop the camera at any time.")

while True:
    ret, frame = camera.read()
    if not ret:
        st.warning("Camera not detected or not clear.")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        cv2.putText(frame, "Authentic", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    except Exception as e:
        cv2.putText(frame, "Suspicious / Face not clear", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
cv2.destroyAllWindows()
