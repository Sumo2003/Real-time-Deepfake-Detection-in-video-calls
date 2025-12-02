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
# 🌈 PAGE CONFIG + CUSTOM STYLES
# -------------------------------------------------------
st.set_page_config(page_title="Real-Time Deepfake Detection v6.4", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, rgba(240,240,255,1) 0%, rgba(220,230,255,1) 100%);
        font-family: 'Poppins', sans-serif;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 25px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4A00E0, #8E2DE2);
        color: white;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px rgba(78,84,200,0.6);
    }
    .main-title {
        text-align: center;
        color: black;
        font-weight: 700;
        font-size: 2.2em;
        letter-spacing: 1px;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 🕵️ TITLE
# -------------------------------------------------------
st.markdown("<h1 class='main-title'>🕵️ Real-Time Deepfake Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Glassmorphism Edition — Real-Time Analysis, Liveness Check & PDF Reporting</p>", unsafe_allow_html=True)

# -------------------------------------------------------
# ⚙️ SIDEBAR SETTINGS
# -------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3433/3433118.png", width=100)
    st.title("⚙️ Settings Panel")
    threshold = st.slider("Similarity Threshold", 0.4, 0.9, 0.60, 0.01)
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
# 🧩 FUNCTIONS
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
    data = {"Time": [timestamp], "Confidence (%)": [confidence], "Label": [label], "Mean Liveness Sim": [mean_sim]}
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
with st.container():
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        FRAME_WINDOW = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📈 Detection Analytics")
        status_placeholder = st.empty()
        confidence_placeholder = st.progress(0)
        chart_placeholder = st.empty()
        pie_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

run_live = st.button("▶️ Start Live Detection", key="start_live_v64")

if run_live:
    cap = cv2.VideoCapture(0)
    emb_window = deque(maxlen=LIVENESS_WINDOW)
    confidences, labels = [], []

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
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live Video Feed")

                log_to_csv(time.strftime("%H:%M:%S"), confidence, label, mean_sim)
                confidences.append(confidence)
                labels.append(label.split()[0])

                df_chart = pd.DataFrame({"Frame": range(1, len(confidences)+1), "Confidence": confidences})
                chart_placeholder.line_chart(df_chart, x="Frame", y="Confidence")

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
# 📄 REPORT SECTION
# -------------------------------------------------------
st.markdown("---")
st.markdown("<h3 style='color:black;'>📄 Generate PDF Report</h3>", unsafe_allow_html=True)
if st.button("Generate Detection Report"):
    generate_pdf_report()
