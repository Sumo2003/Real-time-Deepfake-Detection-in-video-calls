# real_time_deepfake_demo.py
# Simple real-time "deepfake detection" prototype using DeepFace embeddings + OpenCV face detection

import os
import cv2
import time
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# ----------------------------
# Config / Hyperparameters
# ----------------------------
REFS_DIR = "refs"
MODEL_NAME = "VGG-Face"      # DeepFace model
DETECTOR_BACKEND = "opencv"  # face detector backend
ENFORCE_DETECTION = False
SIMILARITY_THRESHOLD = 0.60  # similarity >= this -> "Authentic"

# ----------------------------
# Utilities
# ----------------------------
def ensure_refs_dir():
    if not os.path.exists(REFS_DIR):
        os.makedirs(REFS_DIR)

def load_reference_embeddings(model_name=MODEL_NAME):
    embeddings = []
    for fname in os.listdir(REFS_DIR):
        path = os.path.join(REFS_DIR, fname)
        if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png") or fname.lower().endswith(".jpeg")):
            continue
        try:
            objs = DeepFace.represent(img_path=path,
                                     model_name=model_name,
                                     detector_backend=DETECTOR_BACKEND,
                                     enforce_detection=ENFORCE_DETECTION)
            if len(objs) > 0 and "embedding" in objs[0]:
                emb = np.array(objs[0]["embedding"])
                embeddings.append((path, emb))
                print(f"[INFO] Loaded ref: {fname}, emb_len={len(emb)}")
        except Exception as e:
            print(f"[WARN] Could not process {path}: {e}")
    return embeddings

def get_face_embeddings_from_crop(face_bgr, model_name=MODEL_NAME):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    try:
        objs = DeepFace.represent(img_path=face_rgb,
                                 model_name=model_name,
                                 detector_backend=DETECTOR_BACKEND,
                                 enforce_detection=ENFORCE_DETECTION)
        if len(objs) > 0 and "embedding" in objs[0]:
            return np.array(objs[0]["embedding"])
    except:
        return None
    return None

def similarity_score(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    try:
        dist = cosine(emb1, emb2)
        sim = 1 - dist
        return max(0.0, min(1.0, sim))
    except:
        return 0.0

# ----------------------------
# Main loop
# ----------------------------
def main(camera_index=0, threshold=SIMILARITY_THRESHOLD):
    ensure_refs_dir()
    print("[INFO] Computing reference embeddings (this may take a few seconds)...")
    refs = load_reference_embeddings()
    if len(refs) == 0:
        print("[INFO] No reference images found in ./refs. Press 's' while running the demo to save a reference face.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Exiting.")
        return

    last_ref_reload_time = time.time()
    RELOAD_INTERVAL = 10.0

    print("[INFO] Starting webcam. Press 'q' to quit, 's' to save current face to ./refs")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

        label = "No face"
        confidence_pct = 0.0
        color = (0,255,255)

        if len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x,y,w,h) = faces_sorted[0]
            pad = int(0.15 * w)
            x1 = max(0, x-pad)
            y1 = max(0, y-pad)
            x2 = min(frame.shape[1], x+w+pad)
            y2 = min(frame.shape[0], y+h+pad)
            face_crop = frame[y1:y2, x1:x2]

            emb = get_face_embeddings_from_crop(face_crop)

            best_sim = 0.0
            best_ref = None
            for (rpath, remb) in refs:
                sim = similarity_score(emb, remb)
                if sim > best_sim:
                    best_sim = sim
                    best_ref = rpath

            confidence_pct = round(best_sim * 100, 1)
            if best_sim >= threshold:
                label = f"Authentic ({confidence_pct}%)"
                color = (0,200,0)
            else:
                label = f"Suspicious ({confidence_pct}%)"
                color = (0,0,200)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Deepfake Demo (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(faces) > 0:
            timestamp = int(time.time())
            save_path = os.path.join(REFS_DIR, f"ref_{timestamp}.jpg")
            cv2.imwrite(save_path, face_crop)
            print(f"[INFO] Saved reference image {save_path}. Reloading references...")
            refs = load_reference_embeddings()

        if time.time() - last_ref_reload_time > RELOAD_INTERVAL:
            refs = load_reference_embeddings()
            last_ref_reload_time = time.time()

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
