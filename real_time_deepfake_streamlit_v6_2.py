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

# -------------------------------------------------------
# 🔧 PAGE CONFIG & THEME
# -------------------------------------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v6.2", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🕵️ Real-Time Deepfake Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Enhanced Version with Analytics, Liveness Check & Report Generation</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------
# 📊 SIDEBAR CONFIGURATION
# -------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3433/3433118.png", width=120)
st.sidebar.title("⚙️ Controls & Settings")

threshold = st.sidebar.slider("Similarity Threshold", 0.4, 0.9, 0.60, 0.01)
LIVENESS_WINDOW = 10
LIVENESS_SIM_THRESHOLD = 0.995
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
REFS_DIR = "refs"
LOG_FILE = "detection_log.csv"

# -------------------------------------------------------
# 📂 LOAD REFERENCE EMBEDDINGS
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
    st.warning("⚠️ No reference images found in 'refs' folder.")
    st.stop()

# -------------------------------------------------------
# 🧩 HELPER FUNCTIONS
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
# 🧠 LIVE DETECTION SECTION
# -------------------------------------------------------
col1, col2 = st.columns([1.2, 1])
FRAME_WINDOW = col1.image([], caption="Live Video Feed", use_column_width=True)

col2.subheader("📈 Detection Status")
status_placeholder = col2.empty()
confidence_placeholder = col2.progress(0)

run_live = st.button("▶️ Start Live Detection", key="start_live_v62")

if run_live:
    cap = cv2.VideoCapture(0)
    emb_window = deque(maxlen=LIVENESS_WINDOW)
    confidences, labels = [], []
    chart_placeholder = col2.empty()
    pie_placeholder = col2.empty()

    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        st.info("✅ Webcam connected. Press 'Stop Live Detection' to end.")
        stop_button = st.button("⏹️ Stop Live Detection")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera read error.")
                break
            frame = cv2.flip(frame, 1)
            emb = get_face_embedding(frame)

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
                    label = "Suspicious (Static Face)"
                    color = (0, 0, 255)
                elif sim >= threshold:
                    label = f"Authentic ({confidence}%)"
                    color = (0, 255, 0)
                else:
                    label = f"Suspicious ({confidence}%)"
                    color = (0, 0, 255)

                status_placeholder.markdown(f"### 🧾 {label}")
                confidence_placeholder.progress(confidence)

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                log_to_csv(time.strftime("%H:%M:%S"), confidence, label, mean_sim)
                confidences.append(confidence)
                labels.append(label.split()[0])

                # --- Line chart ---
                df_chart = pd.DataFrame({"Frame": range(1, len(confidences)+1), "Confidence": confidences})
                chart_placeholder.line_chart(df_chart, x="Frame", y="Confidence")

                # --- Pie chart ---
                counts = pd.Series(labels).value_counts()
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#E53935"])
                ax.axis("equal")
                pie_placeholder.pyplot(fig)

            if stop_button:
                st.warning("🛑 Live detection stopped.")
                break

        cap.release()
        cv2.destroyAllWindows()

# -------------------------------------------------------
# 🧾 REPORT GENERATION
# -------------------------------------------------------
st.markdown("---")
st.markdown("<h3 style='color:#4B8BBE;'>📄 Generate PDF Report</h3>", unsafe_allow_html=True)
if st.button("Generate Detection Report"):
    generate_pdf_report()
