import streamlit as st
import numpy as np
from services.model_service import load_model
from utils.styles import apply_custom_styles

st.set_page_config(
    page_title="Deteksi Produktivitas Mahasiswa",
    page_icon="🧠",
    layout="wide"
)
apply_custom_styles()

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%); padding: 2.25rem; border-radius: 20px; color: white !important; margin-bottom: 2rem; box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);">
        <h1 class="no-gradient" style="color: white !important; -webkit-text-fill-color: white !important; font-family: 'Outfit'; font-weight: 800; font-size: 2.2rem; margin: 0 0 10px 0; background: none !important; line-height: 1.2;">
            🧠 Deteksi Produktivitas & Burnout Mahasiswa
        </h1>
        <p style="font-size: 1.05rem; opacity: 0.95; margin: 0; font-family: 'Plus Jakarta Sans'; line-height: 1.6; color: white !important; -webkit-text-fill-color: white !important;">
            Aplikasi analisis cerdas berbasis <b>Jaringan Syaraf Tiruan (JST)</b>. Menggunakan kombinasi algoritma
            <b>Perceptron</b> (klasifikasi terarah) dan <b>Kohonen Self-Organizing Map (SOM)</b> (pengelompokkan mandiri)
            untuk memetakan serta mendeteksi kondisi aktivitas dan kesehatan harian Anda.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="custom-card">
            <h3 style="color: #4F46E5; display: flex; align-items: center; gap: 8px; font-size: 1.2rem;">📌 Analisis Harian</h3>
            <p>
                Masukkan data aktivitas harian Anda (jam tidur, mood, tingkat stres, jam belajar, screen-time, tugas) untuk mendapatkan deteksi instan dan rekomendasi aktivitas dari model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div class="custom-card">
            <h3 style="color: #7C3AED; display: flex; align-items: center; gap: 8px; font-size: 1.2rem;">📊 Dataset Akademik</h3>
            <p>
                Lihat dan telusuri dataset latih mahasiswa yang digunakan untuk mengoptimalkan bobot model klasifikasi Perceptron dan centroid klaster Kohonen.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <div class="custom-card">
            <h3 style="color: #EC4899; display: flex; align-items: center; gap: 8px; font-size: 1.2rem;">📉 Model & Training</h3>
            <p>
                Pantau kurva pembelajaran model JST, rincian bobot akhir untuk setiap fitur harian, posisi koordinat pusat cluster, dan visualisasi pemisahan data latih.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

repo, perceptron, som = load_model()

# Hitung jumlah anggota klaster SOM secara dinamis
clusters = som.predict(repo.X_norm)
nama_klaster = [som.get_cluster_name(c) for c in clusters]
count_produktif = nama_klaster.count("Produktif")
count_santai = nama_klaster.count("Santai")
count_burnout = nama_klaster.count("Burnout")

col_summary_1, col_summary_2 = st.columns(2)

with col_summary_1:
    st.subheader("📋 Label Target Dataset (Perceptron)")
    st.caption("Digunakan untuk model klasifikasi terbimbing Perceptron")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{len(repo.df)} Baris")
    c2.metric("Produktif (Label 1)", f"{int((repo.y == 1).sum())}")
    c3.metric("Tidak Produktif (Label 0)", f"{int((repo.y == 0).sum())}")

with col_summary_2:
    st.subheader("🧩 Hasil Clustering JST (Kohonen SOM)")
    st.caption("Dikelompokkan secara unsupervised oleh jaringan Kohonen")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Klaster Produktif", f"{count_produktif}")
    cc2.metric("Klaster Santai", f"{count_santai}")
    cc3.metric("Klaster Burnout", f"{count_burnout}")

st.divider()