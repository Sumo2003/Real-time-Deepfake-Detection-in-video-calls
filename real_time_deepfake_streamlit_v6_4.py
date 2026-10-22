import io
import os
import time
import threading
from collections import deque

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from deepface import DeepFace
from scipy.spatial.distance import cosine
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v6.4", layout="wide")

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
h1 { text-align: center; color: black; font-size: 2.2em; letter-spacing: 0.5px; }
h3 { color: black; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🕵️ Real-Time Deepfake Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#222;'>Enhanced with Animated UI, Analytics & PDF Report</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------------------------------
# SETTINGS
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
# REFERENCE IMAGES
# -------------------------------------------------------
refs = []
reference_load_error = None

if os.path.isdir(REFS_DIR):
    for img_name in os.listdir(REFS_DIR):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(REFS_DIR, img_name)
            try:
                result = DeepFace.represent(
                    img_path=image_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=ENFORCE_DETECTION
                )
                if result:
                    refs.append(np.array(result[0]["embedding"], dtype=np.float32))
            except Exception as exc:
                reference_load_error = str(exc)

if not refs:
    st.warning(
        "⚠️ Reference images are not available in the app's 'refs' folder. "
        "The webcam preview can run, but similarity-based Authentic/Suspicious "
        "classification is disabled until valid reference images are added."
    )
    if reference_load_error:
        st.caption(f"Reference image loading detail: {reference_load_error[:250]}")

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def get_similarity(embedding):
    if not refs:
        return None
    scores = [1 - cosine(embedding, ref) for ref in refs]
    valid_scores = [float(score) for score in scores if np.isfinite(score)]
    return max(valid_scores) if valid_scores else None


def get_face_embedding(frame):
    try:
        result = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION
        )
        if result:
            return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception:
        return None
    return None


def log_to_csv(timestamp, confidence, label, mean_sim):
    row = pd.DataFrame([{
        "Time": timestamp,
        "Confidence (%)": confidence if confidence is not None else "",
        "Label": label,
        "Mean Liveness Sim": mean_sim if mean_sim is not None else ""
    }])
    header_needed = not os.path.exists(LOG_FILE)
    row.to_csv(LOG_FILE, mode="a", header=header_needed, index=False)


# -------------------------------------------------------
# WEBRTC VIDEO PROCESSOR
# -------------------------------------------------------
class DeepfakeProcessor(VideoProcessorBase):
    def __init__(self):
        self.emb_window = deque(maxlen=LIVENESS_WINDOW)
        self.lock = threading.Lock()
        self.latest = {
            "label": "Waiting for camera...",
            "confidence": None,
            "mean_sim": None,
            "similarity": None,
            "frames": 0,
            "authentic": 0,
            "suspicious": 0,
        }
        self._last_log_time = 0.0

    def get_status(self):
        with self.lock:
            return dict(self.latest)

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)

        label = "No face detected"
        confidence = None
        mean_sim = None
        similarity = None
        color = (0, 165, 255)

        embedding = get_face_embedding(frame)

        if embedding is not None:
            self.emb_window.append(embedding)

            if len(self.emb_window) >= 2:
                similarities = [
                    1 - cosine(first, second)
                    for first, second in zip(
                        list(self.emb_window)[:-1],
                        list(self.emb_window)[1:]
                    )
                ]
                finite_sims = [float(x) for x in similarities if np.isfinite(x)]
                mean_sim = float(np.mean(finite_sims)) if finite_sims else None
                is_live = mean_sim is None or mean_sim <= LIVENESS_SIM_THRESHOLD
            else:
                is_live = True

            similarity = get_similarity(embedding)

            if not refs:
                label = "Reference images required"
                color = (0, 165, 255)
            elif similarity is None:
                label = "Unable to calculate similarity"
                color = (0, 165, 255)
            else:
                confidence = max(0, min(100, int(similarity * 100)))
                if not is_live:
                    label = "Suspicious (Static Face)"
                    color = (0, 0, 255)
                elif similarity >= threshold:
                    label = f"Authentic ({confidence}%)"
                    color = (0, 200, 0)
                else:
                    label = f"Suspicious ({confidence}%)"
                    color = (0, 0, 255)

        cv2.putText(
            frame, label[:65], (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )

        with self.lock:
            self.latest["label"] = label
            self.latest["confidence"] = confidence
            self.latest["mean_sim"] = mean_sim
            self.latest["similarity"] = similarity
            self.latest["frames"] += 1
            if label.startswith("Authentic"):
                self.latest["authentic"] += 1
            elif label.startswith("Suspicious"):
                self.latest["suspicious"] += 1

        # Avoid writing to CSV on every video frame.
        now = time.time()
        if now - self._last_log_time >= 2.0:
            try:
                log_to_csv(
                    time.strftime("%H:%M:%S"),
                    confidence,
                    label,
                    mean_sim
                )
            except Exception:
                pass
            self._last_log_time = now

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# -------------------------------------------------------
# MAIN LAYOUT AND BROWSER WEBCAM
# -------------------------------------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📹 Live Webcam")
    st.caption("Click START below and allow camera permission in your browser.")
    ctx = webrtc_streamer(
        key="deepfake-detection-v2",
        video_processor_factory=DeepfakeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📈 Detection Status")
    if ctx.state.playing and ctx.video_processor:
        status = ctx.video_processor.get_status()
        st.markdown(f"### 🧾 {status['label']}")
        if status["confidence"] is not None:
            st.progress(status["confidence"] / 100)
            st.metric("Similarity confidence", f"{status['confidence']}%")
        else:
            st.caption("A confidence score is available only when reference images load.")
        st.metric("Frames processed", status["frames"])
        metric1, metric2 = st.columns(2)
        metric1.metric("Authentic labels", status["authentic"])
        metric2.metric("Suspicious labels", status["suspicious"])
        if not refs:
            st.info("Add the original reference images to the 'refs' folder to enable similarity classification.")
    else:
        st.info("Press START in the webcam panel to begin. Your browser may ask for camera permission.")

# -------------------------------------------------------
# CHARTS FROM SAVED LOG
# -------------------------------------------------------
st.markdown("---")
st.subheader("📊 Detection Analytics")

if os.path.exists(LOG_FILE):
    try:
        log_df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
        if not log_df.empty:
            if "Confidence (%)" in log_df.columns:
                confidence_df = pd.to_numeric(log_df["Confidence (%)"], errors="coerce")
                valid_confidence = confidence_df.dropna()
                if not valid_confidence.empty:
                    st.line_chart(
                        pd.DataFrame({
                            "Record": range(1, len(valid_confidence) + 1),
                            "Confidence (%)": valid_confidence.to_numpy()
                        }).set_index("Record")
                    )

            if "Label" in log_df.columns:
                counts = log_df["Label"].astype(str).value_counts()
                if not counts.empty:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
                    ax.axis("equal")
                    st.pyplot(fig)
                    plt.close(fig)
    except Exception as exc:
        st.warning(f"Could not display analytics: {exc}")

# -------------------------------------------------------
# PDF REPORT
# -------------------------------------------------------
def build_pdf_report():
    if not os.path.exists(LOG_FILE):
        return None

    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
    except Exception:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    authentic = df["Label"].astype(str).str.contains("Authentic", na=False).sum() if "Label" in df else 0
    suspicious = df["Label"].astype(str).str.contains("Suspicious", na=False).sum() if "Label" in df else 0

    elements = [
        Paragraph("Real-Time Deepfake Detection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Total log records: {len(df)}", styles["Normal"]),
        Paragraph(f"Authentic records: {authentic}", styles["Normal"]),
        Paragraph(f"Suspicious records: {suspicious}", styles["Normal"]),
        Spacer(1, 12),
    ]

    if not df.empty:
        table_data = [df.columns.astype(str).tolist()]
        for row in df.fillna("").astype(str).values.tolist():
            table_data.append(row)
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


st.markdown("---")
st.markdown("<h3>📄 Generate Detection Report</h3>", unsafe_allow_html=True)

if st.button("Generate Detection Report"):
    pdf_bytes = build_pdf_report()
    if pdf_bytes is None:
        st.error("No detection log found yet. Start the webcam first and let it process some frames.")
    else:
        st.success("PDF report generated successfully.")
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name="Deepfake_Report.pdf",
            mime="application/pdf"
        )

   
