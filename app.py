import streamlit as st
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

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

st.title("Deepfake Detection Web App")

st.write("Upload gambar untuk mendeteksi apakah gambar tersebut deepfake atau asli.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

def preprocess_image(img: Image.Image, target_size=(224,224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalisasi jika model butuh
    return img_array

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diupload', use_column_width=True)
    
    # Preprocess dan prediksi
    input_arr = preprocess_image(img)
    pred = model.predict(input_arr)[0][0]  # asumsikan output sigmoid 0-1
    
    # Tampilkan hasil prediksi
    if pred > 0.5:
        st.success(f"Prediksi: Deepfake dengan confidence {pred:.2f}")
    else:
        st.info(f"Prediksi: Asli dengan confidence {1-pred:.2f}")
