import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import requests

# ----------------------------
# Config
# ----------------------------
MODEL_URL = "https://github.com/Abimuruga/Cricket-shot__classifier/releases/download/v1.0/cricket_model.h5"
MODEL_PATH = "cricket_model.h5"
LABELS_PATH = "labels.json"

# ----------------------------
# Download model if not present
# ----------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model..."):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# ----------------------------
# Load class labels
# ----------------------------
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title="Cricket Shot Classifier", page_icon="🏏", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #F2F4F4;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------
# App Content
# ----------------------------
st.markdown("<h1 class='main-title'>🏏 Cricket Shot Classifier</h1>", unsafe_allow_html=True)
st.write("Upload a cricket shot image and let AI identify the type of shot!")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))  # resize as per training
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Display results
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader(f"🔎 Predicted Shot: {class_names[str(predicted_class)]}")
    st.write(f"📊 Confidence: **{confidence*100:.2f}%**")
    st.markdown("</div>", unsafe_allow_html=True)
