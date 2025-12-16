# realtime_mask_mediapipe.py (FINAL FIXED VERSION)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
from collections import deque

# ------------------ SETTINGS ------------------
IMG_SIZE = (128, 128)     # MUST match training size
THRESHOLD = 0.55          # adaptive threshold
SMOOTHING_WINDOW = 3      # rolling window
pred_smooth = deque(maxlen=SMOOTHING_WINDOW)

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("mask_model.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ------------------ MEDIAPIPE ------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ------------------ CLAHE ------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# ------------------ WEBCAM ------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb)

    if result.detections:
        for det in result.detections:
            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            margin = 30
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + bw + margin)
            y2 = min(h, y + bh + margin)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ---------- Preprocess ----------
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_clahe = clahe.apply(face_gray)
            face_rgb = cv2.cvtColor(face_clahe, cv2.COLOR_GRAY2RGB)

            face_resized = cv2.resize(face_rgb, IMG_SIZE)
            face_norm = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_norm, 0)

            # ---------- Predict ----------
            pred = float(model.predict(face_input, verbose=0)[0][0])

            pred_smooth.append(pred)
            pred_avg = sum(pred_smooth) / len(pred_smooth)

            # ---------- Logic ----------
            if pred_avg > THRESHOLD:
                label = "with_mask"
                color = (0, 255, 0)
            else:
                label = "without_mask"
                color = (0, 0, 255)

            # ---------- Draw ----------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({pred_avg:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("Real-Time Mask Detection - Improved", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
