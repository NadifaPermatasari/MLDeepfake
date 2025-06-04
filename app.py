import streamlit as st
import requests
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------- STYLING & PAGE CONFIG ----------
st.set_page_config(page_title="Deepfake Image Detector", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        font-size: 2.8em;
        text-align: center;
        color: #2c3e50;
        margin-top: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- DOWNLOAD MODEL ----------
@st.cache_resource
def download_file_from_gdrive(url, output_path):
    if not os.path.exists(output_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return output_path

# Ganti dengan file ID dari model deepfake kamu di Google Drive
model_url = "https://drive.google.com/uc?export=download&id=1OvUKBw5-9ZEpROTCpmXkGwwUVl9qSHuN"
model_path = "model_slim.h5"

download_file_from_gdrive(model_url, model_path)
model = load_model(model_path)

# ---------- HEADER ----------
st.markdown('<div class="title">🔍 Deepfake Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unggah gambar wajah untuk mendeteksi apakah gambar tersebut asli atau deepfake.</div>', unsafe_allow_html=True)

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader("📂 Pilih gambar wajah (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ---------- PREDICTION ----------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang diupload", use_container_width=True)

    input_arr = preprocess_image(img)
    pred = model.predict(input_arr)[0][0]

    st.markdown("---")
    st.subheader("📊 Hasil Deteksi")

    confidence = float(pred) if pred > 0.5 else 1 - float(pred)
    label = "Deepfake" if pred > 0.5 else "Asli"
    emoji = "🚨" if pred > 0.5 else "✅"
    bar_color = "red" if pred > 0.5 else "green"

    st.markdown(f"""
    <div style='
        padding: 15px;
        border-radius: 10px;
        background-color: {'#ffe6e6' if pred > 0.5 else '#e6ffea'};
        border: 1px solid {'#e74c3c' if pred > 0.5 else '#2ecc71'};
        margin-bottom: 20px;
    '>
        <h3 style='color: {"#c0392b" if pred > 0.5 else "#27ae60"}'>{emoji} Prediksi: {label}</h3>
        <p style='font-size: 16px;'>Confidence: <b>{confidence:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence)
    
