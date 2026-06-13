import streamlit as st
import numpy as np
import pandas as pd
from model import load_model
from helper import normalisasi_data_baru, get_saran

st.set_page_config(page_title="Analisis Mahasiswa", page_icon="📌", layout="wide")

st.header("📌 Input Data Harian Mahasiswa")
st.write("Masukkan data aktivitas harian kamu untuk dianalisis oleh Perceptron dan SOM.")

df, fitur, X, y, X_norm, nilai_min, nilai_max, perceptron, som = load_model()

col1, col2 = st.columns(2)

with col1:
    jam_tidur   = st.slider("Jam Tidur", 0, 12, 7)
    mood        = st.slider("Mood Pagi", 1, 10, 7)
    stres       = st.slider("Tingkat Stres", 1, 10, 4)

with col2:
    jam_belajar  = st.slider("Jam Belajar", 0, 12, 4)
    jam_hp       = st.slider("Jam Main HP", 0, 12, 3)
    jumlah_tugas = st.slider("Jumlah Tugas", 0, 10, 2)

data_baru = np.array([[jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]], dtype=float)

if st.button("🔍 Analisis Sekarang", use_container_width=True):
    hasil_p = perceptron.predict(data_baru)[0]
    hasil_perceptron = "Produktif" if hasil_p == 1 else "Tidak Produktif"

    data_baru_norm = normalisasi_data_baru(data_baru, nilai_min, nilai_max)
    cluster = som.predict(data_baru_norm)[0]
    hasil_som = som.get_cluster_name(cluster)

    st.divider()
    st.subheader("✅ Hasil Analisis")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.metric("Hasil Perceptron", hasil_perceptron)
    with col_h2:
        st.metric("Hasil SOM (Cluster)", hasil_som)

    st.info(get_saran(hasil_perceptron, hasil_som))

    st.write("### 📋 Data yang Dianalisis")
    st.dataframe(pd.DataFrame(data_baru, columns=fitur))
