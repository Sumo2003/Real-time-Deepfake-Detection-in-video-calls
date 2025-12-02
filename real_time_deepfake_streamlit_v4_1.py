import streamlit as st
import cv2
import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import time
from collections import deque

# ---------------------------------
# ✅ Page Config
# ---------------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection", layout="wide")
st.title("🕵️ Real-Time Deepfake Detection Demo")
st.markdown("**Status:** Live Detection using your Webcam")

# ---------------------------------
# ✅ UI Placeholders
# ---------------------------------
FRAME_WINDOW = st.image([])
st.sidebar.title("📊 Detection Info")
status_placeholder = st.sidebar.empty()
confidence_placeholder = st.sidebar.progress(0)
st.sidebar.markdown("---")
st.sidebar.markdown("🎯 **Adjust Settings Below**")

# ---------------------------------
# ✅ Parameters
# ---------------------------------
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
DEFAULT_THRESHOLD = 0.60
REFS_DIR = "refs"

threshold = st.sidebar.slider("Similarity Threshold", 0.4, 0.9, DEFAULT_THRESHOLD, 0.01)

# Liveness parameters
LIVENESS_WINDOW = 10       # number of recent embeddings to keep
LIVENESS_SIM_THRESHOLD = 0.995  # if mean consecutive similarity > this -> suspicious (static)

# ---------------------------------
# ✅ Load Reference Embeddings
# ---------------------------------
refs = []
if os.path.exists(REFS_DIR):
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

if len(refs) == 0:
    st.warning("⚠️ No reference images found in 'refs' folder. Please add at least one image first.")
    st.stop()

# ---------------------------------
# ✅ Helper Functions
# ---------------------------------
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

# ---------------------------------
# ✅ Live Detection
# ---------------------------------
run_live = st.button("▶️ Start Live Detection", key="start_live_v4")

if run_live:
    cap = cv2.VideoCapture(0)
    emb_window = deque(maxlen=LIVENESS_WINDOW)

    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        st.info("✅ Webcam connected. Press **Stop Live** to end.")
        stop_button = st.button("⏹️ Stop Live Detection", key="stop_live_v4")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera read error.")
                break

            frame = cv2.flip(frame, 1)
            emb = get_face_embedding(frame)

            if emb is not None:
                emb_window.append(emb)

                # --- Liveness check ---
                is_live = True
                if len(emb_window) >= 2:
                    sims = []
                    prev = emb_window[0]
                    for e in list(emb_window)[1:]:
                        s = 1 - cosine(prev, e)
                        sims.append(s)
                        prev = e
                    mean_sim = np.mean(sims) if sims else 0.0
                    if mean_sim > LIVENESS_SIM_THRESHOLD:
                        is_live = False
                else:
                    mean_sim = None
                    is_live = True

                # --- Similarity check ---
                sim = get_similarity(emb)
                confidence = int(sim * 100)

                # Combine both results
                if not is_live:
                    label = f"Suspicious (Static Face)"
                    color = (0, 0, 255)
                    status_placeholder.error(f"{label} — low liveness")
                elif sim >= threshold:
                    label = f"Authentic ({confidence}%)"
                    color = (0, 255, 0)
                    status_placeholder.success(label)
                else:
                    label = f"Suspicious ({confidence}%)"
                    color = (0, 0, 255)
                    status_placeholder.error(label)

                confidence_placeholder.progress(confidence)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(frame, (50, 100), (frame.shape[1]-50, frame.shape[0]-100), color, 4)

            else:
                status_placeholder.warning("No face detected.")
                confidence_placeholder.progress(0)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Stop button check
            if stop_button:
                st.warning("🛑 Live detection stopped by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
