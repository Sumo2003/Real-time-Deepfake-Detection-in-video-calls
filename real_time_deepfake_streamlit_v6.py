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

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v6", layout="wide")
st.title("🕵️ Real-Time Deepfake Detection (v6)")
st.markdown("**Enhanced version with Logging, Charts & PDF Report Generation**")

# ----------------------------
# Sidebar & UI placeholders
# ----------------------------
FRAME_WINDOW = st.image([])
st.sidebar.title("📊 Detection Info")
status_placeholder = st.sidebar.empty()
confidence_placeholder = st.sidebar.progress(0)
st.sidebar.markdown("---")

threshold = st.sidebar.slider("Similarity Threshold", 0.4, 0.9, 0.60, 0.01)
LIVENESS_WINDOW = 10
LIVENESS_SIM_THRESHOLD = 0.995

MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
REFS_DIR = "refs"
LOG_FILE = "detection_log.csv"

# ----------------------------
# Load reference embeddings
# ----------------------------
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
    st.warning("⚠️ No reference images found in 'refs' folder.")
    st.stop()

# ----------------------------
# Helper functions
# ----------------------------
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

# ----------------------------
# ✅ Fixed PDF Report Function
# ----------------------------
def generate_pdf_report():
    if not os.path.exists(LOG_FILE):
        st.error("No log file found!")
        return

    # --- Safe CSV read ---
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')  # skips bad rows
    except Exception as e:
        st.error(f"Error reading log file: {e}")
        return

    # --- Basic stats ---
    authentic = df[df["Label"].str.contains("Authentic", case=False, na=False)].shape[0]
    suspicious = df[df["Label"].str.contains("Suspicious", case=False, na=False)].shape[0]
    total = len(df)

    # --- Create PDF ---
    doc = SimpleDocTemplate("Deepfake_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("🕵️ Real-Time Deepfake Detection Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"📅 Date Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"🧩 Total Frames Processed: {total}", styles["Normal"]),
        Paragraph(f"✅ Authentic Frames: {authentic}", styles["Normal"]),
        Paragraph(f"⚠️ Suspicious Frames: {suspicious}", styles["Normal"]),
        Spacer(1, 20)
    ]

    # --- Table of logs ---
    try:
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ]))
        elements.append(table)
    except Exception as e:
        elements.append(Paragraph(f"⚠️ Table could not be generated: {e}", styles["Normal"]))

    # --- Build PDF ---
    doc.build(elements)
    st.success("✅ PDF Report generated successfully: Deepfake_Report.pdf")

# ----------------------------
# Live detection
# ----------------------------
run_live = st.button("▶️ Start Live Detection", key="start_live_v6")

if run_live:
    cap = cv2.VideoCapture(0)
    emb_window = deque(maxlen=LIVENESS_WINDOW)
    confidences, labels = [], []

    chart_placeholder = st.empty()
    pie_placeholder = st.empty()

    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        st.info("✅ Webcam connected. Press Stop Live to end.")
        stop_button = st.button("⏹️ Stop Live Detection", key="stop_live_v6")

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
                if len(emb_window) >= 2:
                    sims = [1 - cosine(a, b) for a, b in zip(list(emb_window)[:-1], list(emb_window)[1:])]
                    mean_sim = np.mean(sims)
                    is_live = mean_sim <= LIVENESS_SIM_THRESHOLD
                else:
                    mean_sim, is_live = None, True

                # --- Similarity check ---
                sim = get_similarity(emb)
                confidence = int(sim * 100)

                if not is_live:
                    label = "Suspicious (Static Face)"
                    color = (0, 0, 255)
                elif sim >= threshold:
                    label = f"Authentic ({confidence}%)"
                    color = (0, 255, 0)
                else:
                    label = f"Suspicious ({confidence}%)"
                    color = (0, 0, 255)

                # --- Display & Log ---
                status_placeholder.text(label)
                confidence_placeholder.progress(confidence)
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                log_to_csv(time.strftime("%H:%M:%S"), confidence, label, mean_sim)

                # --- Update Charts ---
                confidences.append(confidence)
                labels.append(label.split()[0])

                df_chart = pd.DataFrame({
                    "Frame": range(1, len(confidences)+1),
                    "Confidence": confidences
                })
                chart_placeholder.line_chart(df_chart, x="Frame", y="Confidence")

                counts = pd.Series(labels).value_counts()
                fig, ax = plt.subplots()
                ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
                pie_placeholder.pyplot(fig)

            if stop_button:
                st.warning("🛑 Live detection stopped.")
                break

        cap.release()
        cv2.destroyAllWindows()

# ----------------------------
# Report Generation
# ----------------------------
st.markdown("---")
if st.button("📄 Generate Detection Report (PDF)"):
    generate_pdf_report()
