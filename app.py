import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import json
from PIL import Image
import mediapipe as mp

# MUST be the first Streamlit command
st.set_page_config(page_title="Mask Detection App", page_icon="😷")

IMG_SIZE = (128, 128)

# ------------------ Load SavedModel ------------------
@st.cache_resource
def load_model():
    model = tf.saved_model.load("mask_model_converted")   # LOAD SavedModel
    infer = model.signatures["serving_default"]           # USE REAL SIGNATURE
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return infer, idx_to_class

infer, idx_to_class = load_model()

# ------------------ Mediapipe ------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ------------------ Prediction using SavedModel Signature ------------------
def predict_tensor(input_tensor):
    # IMPORTANT: Use exact input name from your signature: "input_layer"
    output_dict = infer(input_layer=tf.constant(input_tensor))
    result_tensor = output_dict["output_0"]  # exact output name
    pred_value = float(result_tensor.numpy()[0][0])
    return pred_value


# ------------------ Detection Function ------------------
def detect_and_classify(image_pil):
    img = np.array(image_pil.convert("RGB"))
    h, w, _ = img.shape

    annotated = img.copy()
    detections = []

    results = face_detector.process(img)

    if not results.detections:
        return annotated, detections

    for det in results.detections:
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

        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        clahe_face = clahe.apply(gray)
        rgb_face = cv2.cvtColor(clahe_face, cv2.COLOR_GRAY2RGB)

        resized = cv2.resize(rgb_face, IMG_SIZE)
        normalized = resized.astype("float32") / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        # ------------------- FINAL FIX: correct signature call -------------------
        pred = predict_tensor(input_tensor)
        # ------------------------------------------------------------------------

        label = "with_mask" if pred > 0.5 else "without_mask"
        color = (0, 255, 0) if label == "with_mask" else (255, 0, 0)

        detections.append({"label": label, "score": pred})

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} ({pred:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return annotated, detections


# ------------------ UI ------------------
st.title("😷 Face Mask Detection App (Week 4 - FINAL)")
st.write("Upload an image or use your camera. Model: **SavedModel Signature**, Mediapipe Detection.")

option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])
image = None

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
else:
    camera_img = st.camera_input("Take a picture")
    if camera_img:
        image = Image.open(camera_img)

if image:
    st.subheader("Input Image")
    st.image(image, use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            annotated, preds = detect_and_classify(image)

        st.subheader("Output")
        st.image(annotated, use_column_width=True)

        if preds:
            st.write("### Detection Results:")
            for i, p in enumerate(preds, 1):
                st.write(f"Face {i}: **{p['label']}** (score: `{p['score']:.3f}`)")
        else:
            st.warning("No faces detected.")
