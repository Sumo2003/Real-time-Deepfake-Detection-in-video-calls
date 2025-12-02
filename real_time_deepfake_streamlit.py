import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import os

# ----------------------------
# Config
# ----------------------------
REFS_DIR = "refs"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
SIMILARITY_THRESHOLD = 0.60

# ----------------------------
# Load reference embeddings
# ----------------------------
def load_reference_embeddings():
    embeddings = []
    for fname in os.listdir(REFS_DIR):
        path = os.path.join(REFS_DIR, fname)
        if not fname.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        try:
            objs = DeepFace.represent(img_path=path, model_name=MODEL_NAME,
                                      detector_backend=DETECTOR_BACKEND,
                                      enforce_detection=False)
            if len(objs) > 0 and "embedding" in objs[0]:
                embeddings.append((path, np.array(objs[0]["embedding"])))
        except:
            continue
    return embeddings

def similarity_score(emb1, emb2):
    dist = cosine(emb1, emb2)
    return max(0.0, min(1.0, 1-dist))

# ----------------------------
# Streamlit App
# ----------------------------
st.title("Mini Real-Time Deepfake Detection Demo")
st.write("Green = Authentic, Red = Suspicious")

# Webcam capture
cap = cv2.VideoCapture(0)
refs = load_reference_embeddings()

frame_window = st.image([])

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    label = "No face"
    confidence = 0.0
    color = (0,255,255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")\
                .detectMultiScale(gray, 1.1, 5, minSize=(80,80))

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda x:x[2]*x[3], reverse=True)[0]
        face_crop = frame[y:y+h, x:x+w]

        try:
            emb = DeepFace.represent(face_crop, MODEL_NAME, DETECTOR_BACKEND, enforce_detection=False)[0]["embedding"]
            best_sim = 0.0
            for rpath, remb in refs:
                sim = similarity_score(emb, remb)
                if sim > best_sim:
                    best_sim = sim
            confidence = round(best_sim*100,1)
            if best_sim >= SIMILARITY_THRESHOLD:
                label = f"Authentic ({confidence}%)"
                color = (0,200,0)
            else:
                label = f"Suspicious ({confidence}%)"
                color = (0,0,200)
        except:
            label = "Face not clear"
            color = (0,255,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb, channels="RGB")
