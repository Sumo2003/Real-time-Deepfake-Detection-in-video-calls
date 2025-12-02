import streamlit as st
import cv2
import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from collections import deque
import mediapipe as mp
import time

# ---------------------------------
# ✅ App Config
# ---------------------------------
st.set_page_config(page_title="Deepfake Detection v5", layout="wide")
st.title("🧠 Advanced Real-Time Deepfake Detection (v5)")
st.markdown("This build integrates multi-model fusion and liveness cues using facial motion tracking.")

FRAME_WINDOW = st.image([])
st.sidebar.title("📊 Detection Metrics")
status_placeholder = st.sidebar.empty()
confidence_placeholder = st.sidebar.progress(0)
st.sidebar.markdown("---")

# ---------------------------------
# ✅ Parameters
# ---------------------------------
MODELS = ["VGG-Face", "Facenet512"]
DETECTOR_BACKEND = "opencv"
REFS_DIR = "refs"
ENFORCE_DETECTION = False
THRESHOLD = 0.60
LIVENESS_WINDOW = 8
LIVENESS_SIM_THRESHOLD = 0.995

# ---------------------------------
# ✅ Load Reference Embeddings (Multi-model fusion)
# ---------------------------------
def load_reference_embeddings():
    refs = []
    if not os.path.exists(REFS_DIR):
        os.makedirs(REFS_DIR)
    for img_name in os.listdir(REFS_DIR):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(REFS_DIR, img_name)
            embs = []
            for model in MODELS:
                try:
                    rep = DeepFace.represent(img_path=path, model_name=model, detector_backend=DETECTOR_BACKEND, enforce_detection=ENFORCE_DETECTION)[0]['embedding']
                    embs.append(np.array(rep))
                except:
                    pass
            if embs:
                fused_emb = np.mean(embs, axis=0)
                refs.append(fused_emb)
    return refs

refs = load_reference_embeddings()
if len(refs) == 0:
    st.warning("⚠️ No reference images found in 'refs' folder. Add at least one.")
    st.stop()

# ---------------------------------
# ✅ Helper Functions
# ---------------------------------
def get_face_embedding(frame):
    try:
        embs = []
        for model in MODELS:
            objs = DeepFace.represent(
                img_path=frame,
                model_name=model,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=ENFORCE_DETECTION
            )
            embs.append(np.array(objs[0]["embedding"]))
        return np.mean(embs, axis=0)
    except:
        return None

def get_similarity(emb):
    if len(refs) == 0:
        return 0.0
    scores = [1 - cosine(emb, r) for r in refs]
    return max(scores)

# ---------------------------------
# ✅ Initialize MediaPipe Face Mesh (for blink detection)
# ---------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]

def calculate_blink_ratio(landmarks, frame_w, frame_h):
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
    right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
    def eye_ratio(eye):
        v = dist(eye[1], eye[2])
        h = dist(eye[0], eye[3])
        return v / (h + 1e-6)
    return (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2

# ---------------------------------
# ✅ Streamlit Controls
# ---------------------------------
run_live = st.button("▶️ Start Live Detection", key="start_live_v5")
stop_live = False

if run_live:
    cap = cv2.VideoCapture(0)
    emb_window = deque(maxlen=LIVENESS_WINDOW)
    blink_counter = 0
    last_blink_time = time.time()

    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        st.info("✅ Webcam connected. Press **Stop Live** to end.")
        stop_button = st.button("⏹️ Stop Live Detection", key="stop_live_v5")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera read error.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Liveness detection via eye blink ---
            blink_ratio = None
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ih, iw, _ = frame.shape
                    landmarks = [(lm.x * iw, lm.y * ih) for lm in face_landmarks.landmark]
                    blink_ratio = calculate_blink_ratio(landmarks, iw, ih)
                    if blink_ratio < 0.20:  # threshold for blink
                        blink_counter += 1
                        last_blink_time = time.time()

            # --- Face embedding similarity ---
            emb = get_face_embedding(frame)
            if emb is not None:
                emb_window.append(emb)

                # Temporal similarity (static face check)
                is_live = True
                if len(emb_window) >= 2:
                    sims = [1 - cosine(emb_window[i], emb_window[i+1]) for i in range(len(emb_window)-1)]
                    mean_sim = np.mean(sims)
                    if mean_sim > LIVENESS_SIM_THRESHOLD:
                        is_live = False
                else:
                    mean_sim = 0.0

                # Blink activity check (liveness)
                blink_live = (time.time() - last_blink_time) < 4  # blink within last 4s
                liveness_score = (0.5 * (1 if is_live else 0)) + (0.5 * (1 if blink_live else 0))

                # Overall decision
                sim = get_similarity(emb)
                confidence = int(sim * 100)

                if not is_live and not blink_live:
                    label = "⚠️ Deepfake Suspected (No motion or blink)"
                    color = (0, 0, 255)
                    status_placeholder.error(label)
                elif sim >= THRESHOLD:
                    label = f"Authentic ({confidence}%)"
                    color = (0, 255, 0)
                    status_placeholder.success(label)
                else:
                    label = f"Suspicious ({confidence}%)"
                    color = (0, 165, 255)
                    status_placeholder.warning(label)

                confidence_placeholder.progress(confidence)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(frame, (50, 100), (frame.shape[1]-50, frame.shape[0]-100), color, 4)
            else:
                status_placeholder.warning("No face detected.")
                confidence_placeholder.progress(0)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if stop_button:
                st.warning("🛑 Live detection stopped by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
