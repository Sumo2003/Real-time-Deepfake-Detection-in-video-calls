import streamlit as st
import cv2
import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import tempfile
import time
import csv
from datetime import datetime

st.set_page_config(page_title="Real-Time Deepfake Detection", layout="wide")

st.title("🕵️ Real-Time Deepfake Detection Demo")
st.markdown("**Status:** Live Detection using your Webcam")

FRAME_WINDOW = st.image([])
st.sidebar.title("📊 Detection Info")
status_placeholder = st.sidebar.empty()
confidence_placeholder = st.sidebar.progress(0)

# Parameters
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
THRESHOLD = 0.60

# Load reference image(s)
REFS_DIR = "refs"
refs = []
for img_name in os.listdir(REFS_DIR):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(REFS_DIR, img_name)
        emb = DeepFace.represent(
            img_path=path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION
        )[0]["embedding"]
        refs.append(np.array(emb))

def get_similarity(emb):
    if len(refs) == 0:
        return 0.0
    scores = [1 - cosine(emb, r) for r in refs]
    return max(scores)

def get_face_embedding(frame):
    try:
        objs = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION
        )
        return np.array(objs[0]["embedding"])
    except:
        return None

# Create or load CSV log
log_file = "detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Result", "Confidence"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Cannot open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera read error.")
            break

        frame = cv2.flip(frame, 1)
        emb = get_face_embedding(frame)

        if emb is not None:
            sim = get_similarity(emb)
            confidence = int(sim * 100)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if sim >= THRESHOLD:
                result = f"Authentic"
                label = f"{result} ({confidence}%)"
                color = (0, 255, 0)
                status_placeholder.success(label)
            else:
                result = f"Suspicious"
                label = f"{result} ({confidence}%)"
                color = (0, 0, 255)
                status_placeholder.error(label)

            confidence_placeholder.progress(confidence)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (50, 100), (frame.shape[1]-50, frame.shape[0]-100), color, 4)

            # --- Save each result to CSV ---
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, result, f"{confidence}%"])
        else:
            status_placeholder.warning("No face detected.")
            confidence_placeholder.progress(0)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
