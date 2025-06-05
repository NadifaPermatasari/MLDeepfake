import streamlit as st
import requests
import tensorflow as tf
import os
import numpy as np
from PIL import Image

# ---------- KONFIGURASI HALAMAN & STYLING (Fokus pada Latar Belakang dan Tata Letak) ----------
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="centered", # Mengatur layout ke 'centered' agar konten di tengah
    initial_sidebar_state="collapsed" # Menyembunyikan sidebar secara default
)

# Custom CSS untuk tampilan latar belakang dan tata letak
st.markdown("""
    <style>
    /* Latar Belakang Utama Aplikasi */
    .stApp {
        background-color: #f0f2f5; /* Warna abu-abu terang yang lembut */
        background-image: url("https://www.transparenttextures.com/patterns/clean-textile.png"); /* Opsi: tambahkan tekstur lembut */
        background-attachment: fixed; /* Membuat latar belakang tetap saat di-scroll */
        background-size: cover; /* Pastikan gambar latar menutupi seluruh area */
    }

    /* Kontainer Utama Konten */
    .block-container {
        max-width: 800px; /* Batasi lebar konten agar tidak terlalu lebar di layar besar */
        padding-top: 3rem; /* Padding atas lebih besar */
        padding-bottom: 3rem; /* Padding bawah lebih besar */
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #ffffff; /* Latar belakang putih untuk area konten utama */
        border-radius: 12px; /* Sudut membulat */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15); /* Bayangan yang lebih menonjol */
        margin-left: auto; /* Pusatkan secara horizontal */
        margin-right: auto; /* Pusatkan secara horizontal */
    }

    /* Header Aplikasi */
    .title {
        font-size: 3.2em; /* Ukuran font lebih besar */
        text-align: center;
        color: #1a2a4b; /* Biru gelap keunguan untuk judul */
        margin-top: 15px;
        margin-bottom: 10px;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1); /* Bayangan teks ringan */
    }
    .subtitle {
        text-align: center;
        font-size: 1.15em;
        color: #555555;
        margin-bottom: 40px; /* Spasi lebih besar setelah subtitle */
        line-height: 1.6;
    }

    /* Styling Umum untuk Elemen Streamlit (misalnya, teks default, label) */
    .stText, .stMarkdown, .stSubheader {
        color: #333333; /* Warna teks gelap yang mudah dibaca */
    }

    /* File Uploader Styling */
    .stFileUploader label {
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #a0a0a0; /* Border abu-abu dashed */
        border-radius: 10px;
        padding: 25px; /* Padding lebih besar */
        background-color: #fcfcfc; /* Latar belakang lebih terang */
        color: #666666;
        transition: all 0.3s ease;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #007bff; /* Biru pada hover */
        background-color: #e6f7ff; /* Latar belakang lebih terang pada hover */
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] svg {
        color: #888888; /* Warna ikon yang lebih lembut */
    }

    /* Tombol */
    .stButton>button {
        background-color: #007bff; /* Biru cerah */
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        cursor: pointer;
        font-size: 1.1em;
        width: 100%; /* Tombol lebar penuh */
        margin-top: 20px; /* Spasi atas */
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Biru lebih gelap saat hover */
        transform: translateY(-2px); /* Efek angkat */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Hasil Prediksi */
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        text-align: center;
        border: 2px solid;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-out; /* Animasi fade-in */
    }
    .prediction-header {
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .confidence-text {
        font-size: 1.4em;
        font-weight: bold;
        margin-top: 15px;
        color: #333;
    }

    /* Expander */
    .stExpander div[data-testid="stExpanderTitle"] {
        font-weight: bold;
        color: #007bff;
        font-size: 1.1em;
    }
    .stExpander div[data-testid="stExpanderContent"] {
        background-color: #f8f9fa; /* Latar belakang abu-abu sangat terang */
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e9ecef;
    }

    /* Progress Bar */
    .stProgress .st-cr {
        background-color: #e0e0e0; /* Warna dasar abu-abu terang */
        height: 10px; /* Tinggi progress bar */
        border-radius: 5px;
    }
    .stProgress .st-cf {
        background-color: #28a745; /* Default green (akan diubah dinamis) */
        border-radius: 5px;
        transition: width 0.5s ease-in-out; /* Animasi perubahan lebar */
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 60px; /* Spasi lebih besar ke footer */
        font-size: 0.9em;
        color: #7f8c8d;
        padding-top: 20px;
        border-top: 1px solid #eeeeee; /* Garis pemisah lembut */
    }

    /* Animasi */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
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
model = tf.keras.models.load_model(model_path)  # ✅ hanya pakai tensorflow

# ---------- HEADER ----------
st.markdown('<div class="title">🔍 Deepfake Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unggah gambar wajah untuk mendeteksi apakah gambar tersebut asli atau deepfake.</div>', unsafe_allow_html=True)

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader("📂 Pilih gambar wajah (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
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

# ---------- FOOTER / BRANDING ----------
st.markdown("""
    <div class="footer">
        <p>By ❤️ Kelompok 5</p>
        <p>Machine Learning</p>   
        <p>Departemen Statistika Bisnis</p>
        <p>Institut Teknologi Sepuluh Nopember (ITS)</p>
    </div>
""", unsafe_allow_html=True)