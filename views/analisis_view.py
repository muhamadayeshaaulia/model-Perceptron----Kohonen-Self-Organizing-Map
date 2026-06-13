import streamlit as st
import numpy as np
import pandas as pd
from utils.helper import normalisasi_data_baru, get_saran


def render(repo, perceptron, som):
    """Render halaman input & analisis produktivitas mahasiswa."""
    st.header("📌 Input Data Harian Mahasiswa")
    st.write("Masukkan data aktivitas harian kamu untuk dianalisis oleh Perceptron dan SOM.")

    col1, col2 = st.columns(2)
    with col1:
        jam_tidur    = st.number_input("🛌 Jam Tidur (jam)", min_value=0, max_value=12, value=7, step=1)
        mood         = st.number_input("😊 Mood Pagi (1–10)", min_value=1, max_value=10, value=7, step=1)
        stres        = st.number_input("😤 Tingkat Stres (1–10)", min_value=1, max_value=10, value=4, step=1)
    with col2:
        jam_belajar  = st.number_input("📚 Jam Belajar (jam)", min_value=0, max_value=12, value=4, step=1)
        jam_hp       = st.number_input("📱 Jam Main HP (jam)", min_value=0, max_value=12, value=3, step=1)
        jumlah_tugas = st.number_input("📝 Jumlah Tugas", min_value=0, max_value=10, value=2, step=1)

    data_baru = np.array(
        [[jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]],
        dtype=float
    )

    if st.button("🔍 Analisis Sekarang", use_container_width=True):
        # Prediksi Perceptron
        hasil_p = perceptron.predict(data_baru)[0]
        hasil_perceptron = "Produktif" if hasil_p == 1 else "Tidak Produktif"

        # Prediksi SOM
        data_baru_norm = normalisasi_data_baru(data_baru, repo.nilai_min, repo.nilai_max)
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
        st.dataframe(pd.DataFrame(data_baru, columns=repo.fitur), width="stretch")
