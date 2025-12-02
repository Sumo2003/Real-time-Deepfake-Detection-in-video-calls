import cv2
from deepface import DeepFace
import os

REFS_DIR = "refs"
MODEL = "VGG-Face"
DETECTOR = "opencv"

# Load all reference images
refs = [os.path.join(REFS_DIR, f) for f in os.listdir(REFS_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if len(refs) == 0:
    print("❌ No reference images found in 'refs' folder. Run your previous script and press 'S' to save one.")
    exit()

ref_path = refs[0]
print(f"[INFO] Using reference image: {ref_path}")

# Compare live webcam face
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("[INFO] Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    try:
        result = DeepFace.verify(img1_path=frame, img2_path=ref_path, 
                                 model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
        verified = result["verified"]
        confidence = round((1 - result["distance"]) * 100, 2)

        text = f"{'Authentic' if verified else 'Suspicious'} ({confidence}%)"
        color = (0, 255, 0) if verified else (0, 0, 255)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Compare Faces Demo", frame)

    except Exception as e:
        cv2.putText(frame, "Face not detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Compare Faces Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
