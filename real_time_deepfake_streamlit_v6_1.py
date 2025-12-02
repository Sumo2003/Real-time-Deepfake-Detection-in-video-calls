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
import base64
import time

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v6.1", layout="wide")
st.title("🕵️ Real-Time Deepfake Detection (v6.1)")
st.markdown("### Enhanced UI • Live Analytics • Auto Report Generation")

# ----------------------------
# GLOBAL SETTINGS
# ----------------------------
FRAME_WINDOW = st.image([])
st.sidebar.title("📊 Detection Dashboard")

status_placeholder = st.sidebar.empty()
confidence_placeholder = st.sidebar.progress(0)
st.sidebar.markdown("---")

threshold = st.sidebar.slider("🧠 Face Similarity Threshold", 0.4, 0.9, 0.60, 0.01)
LIVENESS_WINDOW = 10
LIVENESS_SIM_THRESHOLD = 0.995

MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
REFS_DIR = "refs"
LOG_FILE = "detection_log.csv"

# ----------------------------
# LOAD REFERENCE IMAGES
# ----------------------------
st.sidebar.subheader("📁 Reference Images")
refs = []
if os.path.exists(REFS_DIR):
    imgs = [f for f in os.listdir(REFS_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    st.sidebar.write(f"Loaded {len(imgs)} images from `{REFS_DIR}`")
    for img_name in imgs:
        emb = DeepFace.represent(
            img_path=os.path.join(REFS_DIR, img_name),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION
        )[0]["embedding"]
        refs.append(np.array(emb))
else:
    st.warning("⚠️ No reference folder found! Please create a `refs` folder with authentic images.")
    st.stop()

if len(refs) == 0:
    st.warning("⚠️ No valid reference images found in `refs` folder.")
    st.stop()

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_similarity(emb):
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
    """Generates detection summary report as PDF"""
    if not os.path.exists(LOG_FILE):
        st.error("❌ No log file found! Run detection first.")
        return None

    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading log file: {e}")
        return None

    authentic = df[df["Label"].str.contains("Authentic")].shape[0]
    suspicious = df[df["Label"].str.contains("Suspicious")].shape[0]
    total = len(df)

    report_path = "Deepfake_Report.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("🕵️ Real-Time Deepfake Detection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"Total Frames Processed: {total}", styles["Normal"]),
        Paragraph(f"Authentic Frames: {authentic}", styles["Normal"]),
        Paragraph(f"Suspicious Frames: {suspicious}", styles["Normal"]),
        Spacer(1, 12),
    ]

    # Table of results
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    elements.append(table)

    doc.build(elements)
    return report_path

def download_link(file_path):
    """Creates a download link for the generated PDF"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">📥 Click here to download your report</a>'
    return href

# ----------------------------
# LIVE DETECTION
# ----------------------------
st.markdown("---")
run_live = st.button("🎥 Start Live Detection")

if run_live:
    cap = cv2.VideoCapture(0)
    emb_window = deque(maxlen=LIVENESS_WINDOW)
    confidences, labels = [], []
    chart_placeholder = st.empty()
    pie_placeholder = st.empty()
    fps_placeholder = st.sidebar.empty()

    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        st.success("✅ Webcam connected! Press 'Stop Live Detection' to end.")
        stop_live = st.button("⛔ Stop Live Detection")

        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera read error.")
                break

            frame = cv2.flip(frame, 1)
            emb = get_face_embedding(frame)
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            fps_placeholder.text(f"📸 FPS: {fps:.2f}")

            if emb is not None:
                emb_window.append(emb)
                if len(emb_window) >= 2:
                    sims = [1 - cosine(a, b) for a, b in zip(list(emb_window)[:-1], list(emb_window)[1:])]
                    mean_sim = np.mean(sims)
                    is_live = mean_sim <= LIVENESS_SIM_THRESHOLD
                else:
                    mean_sim, is_live = None, True

                sim = get_similarity(emb)
                confidence = int(sim * 100)

                if not is_live:
                    label = "Suspicious (Static)"
                    color = (0, 0, 255)
                elif sim >= threshold:
                    label = f"Authentic ({confidence}%)"
                    color = (0, 255, 0)
                else:
                    label = f"Suspicious ({confidence}%)"
                    color = (0, 0, 255)

                status_placeholder.text(label)
                confidence_placeholder.progress(confidence)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                log_to_csv(time.strftime("%H:%M:%S"), confidence, label, mean_sim)

                # Update charts
                confidences.append(confidence)
                labels.append(label.split()[0])
                df_chart = pd.DataFrame({"Frame": range(1, len(confidences)+1), "Confidence": confidences})
                chart_placeholder.line_chart(df_chart, x="Frame", y="Confidence", height=200)

                # Pie chart
                counts = pd.Series(labels).value_counts()
                fig, ax = plt.subplots()
                ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
                pie_placeholder.pyplot(fig)

            if stop_live:
                st.warning("🛑 Detection stopped.")
                break

        cap.release()
        cv2.destroyAllWindows()

# ----------------------------
# REPORT GENERATION
# ----------------------------
st.markdown("---")
st.subheader("📄 Generate and Download Detection Report")

if st.button("🧾 Generate PDF Report"):
    report_file = generate_pdf_report()
    if report_file:
        st.success("✅ PDF report successfully generated!")
        st.markdown(download_link(report_file), unsafe_allow_html=True)
