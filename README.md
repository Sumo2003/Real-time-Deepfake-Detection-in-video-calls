🎭 Real-Time Deepfake Detection in Video Calls
A real-time deepfake detection framework that analyzes live webcam feeds using advanced computer vision and AI models to identify manipulated faces with high precision.

🚀 Overview
Deepfakes pose a serious threat in virtual communication.
This project provides real-time detection of manipulated faces during live video calls using advanced computer vision, ML models, and Streamlit.
The system continuously analyzes webcam frames and raises alerts when it identifies signs of synthetic/manipulated faces.

✨ Features
🔍 Real-time Deepfake Detection (frame-by-frame analysis)
👤 Face Recognition + Comparison Models
📹 Live Webcam Feed Processing
🧠 ML-based Confidence Scoring
📊 Detection History Saved in CSV
⚡ Fast, lightweight, and accurate

🛠️ Tech Stack
Component	Technology
Programming Language	Python
Computer Vision	OpenCV
Machine Learning	CNN-based model
UI Framework	Streamlit
Data Handling	NumPy, Pandas

📂Project Structure
Real-time-Deepfake-Detection/
│── combine_faces_demo.py
│── real_time_deepfake_streamlit_v6_4.py   (recommended version)
│── real_time_deepfake_demo.py
│── detection_log.csv
│── requirements.txt
│── Deepfake_Report.pdf
│── refs/  
│── env/

📦 Installation
pip install -r requirements.txt

▶️ Run the Application
Most stable version:
python real_time_deepfake_streamlit_v6_4.py
Old versions are also available for testing.

📘 How It Works
System reads webcam frames in real time
Face detection using Haar Cascades / DNN
Extracted face is passed to ML model
The system generates a deepfake probability score
If above threshold → alert displayed instantly

🙌 Author
Sumaira Ashfaque
AI & Software Developer
Passionate about cybersecurity, deepfake prevention, and real-time vision systems
