import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from deepface import DeepFace
from scipy.spatial.distance import cosine
from collections import deque
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v6.4", layout="wide")

# Inject animated background and glassmorphism CSS
st.markdown("""
<style>
body {
  margin: 0;
  font-family: "Poppins", sans-serif;
  background: linear-gradient(-45deg, #1e3c72, #2a5298, #6a11cb, #2575fc);
  background-size: 400% 400%;
  animation: gradientShift 12s ease infinite;
}
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
section.main > div {
  background: rgba(255, 255, 255, 0.10);
  backdrop-filter: blur(10px);
  border-radius: 18px;
  padding: 25px;
  box-shadow: 0 4px 30px rgba(0,0,0,0.2);
}
h1 {
  text-align: center;
  color: black;
  font-size: 2.2em;
  letter-spacing: 0.5px;
}
h3 {
  color: black;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🕵️ Real-Time Deepfake Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#222;'>Enhanced with Animated UI, Analytics & PDF Report</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3433/3433118.png", width=110)
st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider("Similarity Threshold", 0.4, 0.9, 0.60, 0.01)
LIVENESS_WINDOW = 10
LIVENESS_SIM_THRESHOLD = 0.995
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
REFS_DIR = "refs"
LOG_FILE = "detection_log.csv"

# -------------------------------------------------------
# LOAD REFERENCE EMBEDDINGS
# -------------------------------------------------------
refs = []
if os.path.exists(REFS_DIR):
    for img_name in os.listdir(REFS_DIR):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            emb = DeepFace.represent(
                img_path=os.path.join(REFS_DIR, img_name),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=ENFORCE_DETECTION
            )[0]["embedding"]
            refs.append(np.array(emb))

if len(refs) == 0:
    st.warning(
        "⚠️ No reference images found in 'refs' folder. "
        "Please add reference images to enable similarity-based detection."
    )
# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------
def get_similarity(emb):
    scores = [1 - cosine(emb, r) for r in refs] if refs else [0]
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

def log_to_csv(timestamp, confidence, label, mean_sim):
    data = {
        "Time": [timestamp],
        "Confidence (%)": [confidence],
        "Label": [label],
        "Mean Liveness Sim": [mean_sim]
    }
    df = pd.DataFrame(data)
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

def generate_pdf_report():
    if not os.path.exists(LOG_FILE):
        st.error("No log file found!")
        return
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading log file: {e}")
        return

    authentic = df[df["Label"].str.contains("Authentic")].shape[0]
    suspicious = df[df["Label"].str.contains("Suspicious")].shape[0]
    total = len(df)

    doc = SimpleDocTemplate("Deepfake_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Real-Time Deepfake Detection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Total Frames Processed: {total}", styles["Normal"]),
        Paragraph(f"Authentic Frames: {authentic}", styles["Normal"]),
        Paragraph(f"Suspicious Frames: {suspicious}", styles["Normal"]),
        Spacer(1, 12)
    ]
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    elements.append(table)
    doc.build(elements)
    st.success("✅ PDF Report generated successfully: Deepfake_Report.pdf")

# -------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------
col1, col2 = st.columns([1.2, 1])

FRAME_WINDOW = col1.empty()
col2.subheader("📈 Detection Status")
status_placeholder = col2.empty()
confidence_placeholder = col2.progress(0)
chart_placeholder = col2.empty()
pie_placeholder = col2.empty()

run_live = st.button("▶️ Start Live Detection")

# -------------------------------------------------------
# LIVE DETECTION
# -------------------------------------------------------
class DeepfakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.emb_window = deque(maxlen=LIVENESS_WINDOW)

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)

        emb = get_face_embedding(frame)

        if emb is not None:
            self.emb_window.append(emb)

            if len(self.emb_window) >= 2:
                sims = [
                    1 - cosine(a, b)
                    for a, b in zip(
                        list(self.emb_window)[:-1],
                        list(self.emb_window)[1:]
                    )
                ]
                mean_sim = float(np.mean(sims))
                is_live = mean_sim <= LIVENESS_SIM_THRESHOLD
            else:
                is_live = True

            sim = get_similarity(emb)
            confidence = max(0, min(100, int(sim * 100)))

            if not is_live:
                label = "Suspicious (Static Face)"
                color = (0, 0, 255)
            elif sim >= threshold:
                label = f"Authentic ({confidence}%)"
                color = (0, 255, 0)
            else:
                label = f"Suspicious ({confidence}%)"
                color = (0, 0, 255)

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        else:
            cv2.putText(
                frame, "No face detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
            )

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# -------------------------------------------------------
# REPORT GENERATION
# -------------------------------------------------------
st.markdown("---")
st.markdown("<h3>📄 Generate Detection Report</h3>", unsafe_allow_html=True)
if st.button("Generate Detection Report"):
    generate_pdf_report()
