import os
import json
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# URLs & local paths
# -----------------------------
MODEL_URL = "https://github.com/Abimuruga/Cricket-shot__classifier/releases/download/v1.0/cricket_model.h5"
LABELS_URL = "https://raw.githubusercontent.com/Abimuruga/Cricket-shot__classifier/main/labels.json"

MODEL_PATH = "cricket_model.h5"
LABELS_PATH = "labels.json"

# -----------------------------
# Helpers
# -----------------------------
def download_file(url: str, path: str):
    """Download a file if it doesn't exist."""
    if os.path.exists(path):
        return
    try:
        st.write(f"📥 Downloading {os.path.basename(path)} ...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success(f"✅ Downloaded {os.path.basename(path)}")
    except Exception as e:
        st.error(f"❌ Failed to download {os.path.basename(path)}: {e}")
        st.stop()

def load_labels(path: str):
    """Load labels from JSON; support list or dict formats."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # If dict like {"0":"...", "1":"..."}, convert to list sorted by numeric key
        if isinstance(data, dict):
            # sort by int(key) to ensure correct order
            items = sorted(data.items(), key=lambda kv: int(kv[0]))
            return [v for _, v in items]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("labels.json must be a list or dict")
    except Exception as e:
        st.error(f"❌ Failed to load labels: {e}")
        st.stop()

@st.cache_resource
def get_model(model_path: str):
    """Load and cache the TensorFlow model."""
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

def preprocess_image(pil_img: Image.Image, model_input_shape):
    """
    Preprocess to match model input.
    model_input_shape is (None, H, W, C) or similar; we extract H, W.
    PIL resize expects (width, height).
    """
    # model.input_shape example: (None, 120, 160, 3)
    if len(model_input_shape) < 4:
        st.error(f"Unexpected model input_shape: {model_input_shape}")
        st.stop()

    H = model_input_shape[1]
    W = model_input_shape[2]
    # Resize with PIL: (width, height)
    img_resized = pil_img.resize((W, H))
    # Normalize to [0,1]
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    # If the model expects 3 channels but the image came as grayscale, expand
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# Ensure files exist
# -----------------------------
download_file(MODEL_URL, MODEL_PATH)
download_file(LABELS_URL, LABELS_PATH)

# -----------------------------
# Load resources
# -----------------------------
labels = load_labels(LABELS_PATH)
model = get_model(MODEL_PATH)

# -----------------------------
# UI
# -----------------------------
st.title("🏏 Cricket Shot Classifier")
st.caption("Upload an image to classify the cricket shot.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")

    # Preprocess using the model's expected input size (e.g., 120x160x3)
    img_array = preprocess_image(img, model.input_shape)
    st.write(f"🖼️ Processed input shape: {img_array.shape}")  # should be (1, H, W, 3)

    # Predict
    try:
        preds = model.predict(img_array)
        # preds shape: (1, num_classes)
        probs = preds[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Display
    if 0 <= idx < len(labels):
        predicted_label = labels[idx]
    else:
        predicted_label = f"Class #{idx}"

    st.subheader("📌 Prediction Result")
    st.write(f"**Class:** {predicted_label}")
    st.write(f"**Confidence:** {conf*100:.2f}%")

    # Optional: show top-3
    if len(labels) >= 3:
        top3_idx = np.argsort(probs)[-3:][::-1]
        st.write("**Top-3 predictions:**")
        for i in top3_idx:
            st.write(f"- {labels[i]} — {probs[i]*100:.2f}%")
