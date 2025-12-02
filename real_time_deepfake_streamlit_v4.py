# real_time_deepfake_streamlit_v4.py
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
from pathlib import Path
from typing import List, Tuple

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v4", layout="wide")
st.title("🕵️ Real-Time Deepfake Detection — v4")
st.markdown("Upload reference faces, manage them, and run live comparison. Cached embeddings for faster runtime.")

REFS_DIR = Path("refs")
REFS_DIR.mkdir(exist_ok=True)
LOG_FILE = "detection_log.csv"

MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
THRESHOLD = 0.60  # tune this (0.55..0.65) based on your tests

# -----------------------
# Utilities
# -----------------------
def save_uploaded_file(uploaded_file, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def list_reference_images() -> List[Path]:
    return [p for p in REFS_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

def compute_embedding_from_path(path_or_image):
    """
    Accepts either a filesystem path (Path/str) or an OpenCV image (numpy array).
    Returns embedding numpy array or None.
    """
    try:
        if isinstance(path_or_image, (str, Path)):
            objs = DeepFace.represent(img_path=str(path_or_image),
                                      model_name=MODEL_NAME,
                                      detector_backend=DETECTOR_BACKEND,
                                      enforce_detection=ENFORCE_DETECTION)
        else:
            # image array
            objs = DeepFace.represent(img_path=path_or_image,
                                      model_name=MODEL_NAME,
                                      detector_backend=DETECTOR_BACKEND,
                                      enforce_detection=ENFORCE_DETECTION)
        if len(objs) > 0 and "embedding" in objs[0]:
            return np.array(objs[0]["embedding"])
    except Exception as e:
        # represent may fail for unclear faces
        return None
    return None

def similarity(emb1, emb2):
    try:
        return float(max(0.0, min(1.0, 1 - cosine(emb1, emb2))))
    except:
        return 0.0

def ensure_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "BestRef", "Result", "Confidence"])

def append_log(best_ref, result, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, best_ref if best_ref else "", result, f"{confidence:.2f}%"])

# -----------------------
# Session cache for reference embeddings
# -----------------------
if "refs_embeddings" not in st.session_state:
    st.session_state.refs_embeddings = {}  # path_str -> embedding (np.array)

def reload_refs_embeddings():
    st.session_state.refs_embeddings = {}
    for p in list_reference_images():
        emb = compute_embedding_from_path(str(p))
        if emb is not None:
            st.session_state.refs_embeddings[str(p)] = emb

# Initialize
ensure_log()
if len(st.session_state.refs_embeddings) == 0:
    reload_refs_embeddings()

# -----------------------
# Sidebar: Reference management
# -----------------------
st.sidebar.header("➕ Reference Management")
with st.sidebar.expander("Upload new reference image"):
    uploaded = st.file_uploader("Choose an image (jpg/png). Best: frontal face", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        save_path = save_uploaded_file(uploaded, REFS_DIR)
        st.success(f"Saved to {save_path.name}")
        # compute embedding and store in session
        emb = compute_embedding_from_path(str(save_path))
        if emb is not None:
            st.session_state.refs_embeddings[str(save_path)] = emb
            st.success("Reference embedding computed and cached.")
        else:
            st.warning("Face embedding could not be computed from uploaded image. Try another clear frontal face.")

st.sidebar.markdown("---")
st.sidebar.subheader("Saved references")
refs_list = list_reference_images()
if len(refs_list) == 0:
    st.sidebar.info("No saved references. Upload one above or use old demo to save via 'S'.")
else:
    for p in refs_list:
        cols = st.sidebar.columns([3,1])
        cols[0].write(p.name)
        if cols[1].button("Delete", key=f"del_{p.name}"):
            # delete file and remove from session cache
            try:
                os.remove(p)
                if str(p) in st.session_state.refs_embeddings:
                    del st.session_state.refs_embeddings[str(p)]
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error("Could not delete file.")

st.sidebar.markdown("---")
st.sidebar.subheader("Options")
threshold = st.sidebar.slider("Similarity threshold", 0.40, 0.90, THRESHOLD, 0.01)
mode = st.sidebar.radio("Mode", ["Live Camera", "Upload Video (file)"])

# -----------------------
# Main area: Detection info
# -----------------------
col1, col2 = st.columns([3,1])
with col2:
    st.info("Detection Info")
    status_placeholder = st.empty()
    conf_placeholder = st.empty()
    st.write("Saved refs:", len(st.session_state.refs_embeddings))

with col1:
    stframe = st.image([])

# Ensure embeddings exist
if len(st.session_state.refs_embeddings) == 0:
    st.warning("No cached reference embeddings available. Upload a reference image in the sidebar.")
    st.stop()

# -----------------------
# Live Camera Mode
# -----------------------
def live_camera_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

    last_time = time.time()
    fps_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # display small note about fps occasionally
            fps_counter += 1
            if time.time() - last_time > 1.0:
                last_time = time.time()
                fps_counter = 0

            # try to get embedding for current frame
            emb = compute_embedding_from_path(frame)
            if emb is not None:
                # find best match
                best_sim = 0.0
                best_ref = None
                for rpath, remb in st.session_state.refs_embeddings.items():
                    sim = similarity(emb, remb)
                    if sim > best_sim:
                        best_sim = sim
                        best_ref = rpath

                confidence = best_sim * 100
                if best_sim >= threshold:
                    result = "Authentic"
                    color = (0, 255, 0)
                    status_placeholder.success(f"{result} ({confidence:.2f}%) — matched {Path(best_ref).name}")
                else:
                    result = "Suspicious"
                    color = (0, 0, 255)
                    status_placeholder.error(f"{result} ({confidence:.2f}%) — best {Path(best_ref).name if best_ref else 'N/A'}")
                conf_placeholder.progress(int(confidence))

                # draw box and label
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (30, 80), (w-30, h-80), color, 3)
                cv2.putText(frame, f"{result} {confidence:.1f}%", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                # log
                append_log(best_ref if best_ref else "", result, confidence)
            else:
                status_placeholder.warning("Face not clear / no face detected.")
                conf_placeholder.progress(0)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Streamlit needs a short sleep, adjust for latency
            if st.button("Stop Live", key="stop_live_v4"):
                break
    finally:
        cap.release()

# -----------------------
# Upload Video Mode
# -----------------------
def process_video_upload(u):
    # save temp file
    tmp = save_uploaded_file(u, Path(tempfile.gettempdir()))
    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        st.error("Cannot open uploaded video.")
        return
    st.info("Processing video — will log results per frame. This may take time.")
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        frame = cv2.flip(frame, 1)
        emb = compute_embedding_from_path(frame)
        if emb is not None:
            best_sim = 0.0
            best_ref = None
            for rpath, remb in st.session_state.refs_embeddings.items():
                sim = similarity(emb, remb)
                if sim > best_sim:
                    best_sim = sim
                    best_ref = rpath
            confidence = best_sim * 100
            result = "Authentic" if best_sim >= threshold else "Suspicious"
            append_log(best_ref if best_ref else "", result, confidence)
        st.write(f"Processed frame {frame_no}", end="\r")
    cap.release()
    st.success("Video processing complete. Check detection_log.csv for frame-wise entries.")

# -----------------------
# Run mode
# -----------------------
if mode == "Live Camera":
    st.success("Running Live Camera — press 'Stop Live' to stop.")
    live_camera_loop()
else:
    uploaded_vid = st.file_uploader("Upload a video file (mp4/avi) to analyze", type=["mp4", "avi", "mov"])
    if uploaded_vid is not None:
        process_video_upload(uploaded_vid)
