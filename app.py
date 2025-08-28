import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import requests

# -----------------------------
# Model + Labels Setup
# -----------------------------
MODEL_URL = "https://github.com/Abimuruga/Cricket-shot__classifier/releases/download/v1.0/cricket_model.h5"
LABELS_URL = "https://raw.githubusercontent.com/Abimuruga/Cricket-shot__classifier/main/labels.json"

MODEL_PATH = "cricket_model.h5"
LABELS_PATH = "labels.json"

# Function to download a file if missing
def download_file(url, path):
    try:
        st.write(f"📥 Downloading {path}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        st.success(f"✅ {path} downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download {path}: {e}")
        st.stop()

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)

# Ensure labels exist
if not os.path.exists(LABELS_PATH):
    download_file(LABELS_URL, LABELS_PATH)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load class labels
try:
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"❌ Failed to load labels: {e}")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏏 Cricket Shot Classifier")
st.write("Upload an image to classify the cricket shot.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    Resize correctly: (width=120, height=160)
    img = img.resize((120, 160))   # not (160,120)

# Normalize
    img_array = np.array(img) / 255.0

# Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    st.write(f"🖼️ Final input shape to model: {img_array.shape}")


    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Output
    st.subheader("📌 Prediction Result")
    st.write(f"**Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")


