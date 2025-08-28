import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

# -----------------------------
# Load class labels
# -----------------------------
LABELS_PATH = "labels.json"
if not os.path.exists(LABELS_PATH):
    st.error(f"❌ '{LABELS_PATH}' not found. Please place it in the same folder as app.py.")
else:
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "cricket_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ '{MODEL_PATH}' not found. Please download it from:\n"
             f"https://github.com/Abimuruga/Cricket-shot__classifier/releases/download/v1.0/cricket_model.h5")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏏 Cricket Shot Classifier")
st.write("Upload an image of a cricket shot, and the model will predict the type of shot!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # Preprocess Image (IMPORTANT: Must match training size!)
    # -----------------------------
    img = load_img(uploaded_file, target_size=(120, 160))  # 🔥 Adjust size if training used different
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.subheader("📌 Prediction Result")
    st.write(f"**Predicted Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

