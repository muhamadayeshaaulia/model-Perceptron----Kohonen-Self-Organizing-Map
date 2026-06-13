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
        jam_tidur    = st.number_input("🛌 Jam Tidur (jam)", value=7, step=1)
        if jam_tidur < repo.nilai_min[0] or jam_tidur > repo.nilai_max[0]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[0])} - {int(repo.nilai_max[0])})</p>", unsafe_allow_html=True)
        mood         = st.number_input("😊 Mood Pagi (1–10)", value=7, step=1)
        if mood < repo.nilai_min[1] or mood > repo.nilai_max[1]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[1])} - {int(repo.nilai_max[1])})</p>", unsafe_allow_html=True)

        stres        = st.number_input("😤 Tingkat Stres (1–10)", value=4, step=1)
        if stres < repo.nilai_min[2] or stres > repo.nilai_max[2]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[2])} - {int(repo.nilai_max[2])})</p>", unsafe_allow_html=True)

    with col2:
        jam_belajar  = st.number_input("📚 Jam Belajar (jam)", value=4, step=1)
        if jam_belajar < repo.nilai_min[3] or jam_belajar > repo.nilai_max[3]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[3])} - {int(repo.nilai_max[3])})</p>", unsafe_allow_html=True)
        jam_hp       = st.number_input("📱 Jam Main HP (jam)", value=3, step=1)
        if jam_hp < repo.nilai_min[4] or jam_hp > repo.nilai_max[4]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[4])} - {int(repo.nilai_max[4])})</p>", unsafe_allow_html=True)
        jumlah_tugas = st.number_input("📝 Jumlah Tugas", value=2, step=1)
        if jumlah_tugas < repo.nilai_min[5] or jumlah_tugas > repo.nilai_max[5]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[5])} - {int(repo.nilai_max[5])})</p>", unsafe_allow_html=True)

    any_out_of_bounds = (
        jam_tidur < repo.nilai_min[0] or jam_tidur > repo.nilai_max[0] or
        mood < repo.nilai_min[1] or mood > repo.nilai_max[1] or
        stres < repo.nilai_min[2] or stres > repo.nilai_max[2] or
        jam_belajar < repo.nilai_min[3] or jam_belajar > repo.nilai_max[3] or
        jam_hp < repo.nilai_min[4] or jam_hp > repo.nilai_max[4] or
        jumlah_tugas < repo.nilai_min[5] or jumlah_tugas > repo.nilai_max[5]
    )

    if any_out_of_bounds:
        st.error("❌ Analisis tidak dapat dilakukan karena ada input di luar batas data latih. Silakan sesuaikan nilai input Anda.")

    data_baru = np.array(
        [[jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]],
        dtype=float
    )

    if st.button("🔍 Analisis Sekarang", use_container_width=True, disabled=any_out_of_bounds):
        # Batasi input baru agar tidak melebihi rentang data latih (mencegah error/bias matematis)
        data_baru_clipped = np.clip(data_baru, repo.nilai_min, repo.nilai_max)

        # Prediksi Perceptron
        hasil_p = perceptron.predict(data_baru_clipped)[0]
        hasil_perceptron = "Produktif" if hasil_p == 1 else "Tidak Produktif"

        # Prediksi SOM
        data_baru_norm = normalisasi_data_baru(data_baru_clipped, repo.nilai_min, repo.nilai_max)
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
