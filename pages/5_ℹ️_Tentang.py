import streamlit as st

st.set_page_config(page_title="Tentang Aplikasi", page_icon="ℹ️", layout="wide")

st.header("ℹ️ Tentang Aplikasi")

st.write("""
Aplikasi ini dibuat untuk proyek UAS mata kuliah **Jaringan Syaraf Tiruan**.
Tujuannya adalah membantu mahasiswa menganalisis kondisi produktivitas dan potensi burnout
berdasarkan pola harian.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Model yang Digunakan")
    st.write("""
    **1. Perceptron**

    Digunakan untuk klasifikasi biner — menentukan apakah kondisi mahasiswa
    termasuk **Produktif** atau **Tidak Produktif**.

    **2. Kohonen Self-Organizing Map (SOM)**

    Digunakan untuk clustering — mengelompokkan data mahasiswa menjadi:
    - 🟢 Produktif
    - 🔵 Santai
    - 🔴 Burnout
    """)

with col2:
    st.subheader("📥 Input Sistem")
    st.write("""
    - Jam tidur
    - Mood pagi
    - Tingkat stres
    - Jam belajar
    - Jam main HP
    - Jumlah tugas
    """)

    st.subheader("📤 Output Sistem")
    st.write("""
    - Hasil klasifikasi Perceptron
    - Hasil clustering SOM
    - Saran aktivitas harian
    """)
