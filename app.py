import streamlit as st
from model import load_model

st.set_page_config(
    page_title="Deteksi Produktivitas Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Deteksi Produktivitas dan Burnout Mahasiswa")
st.write(
    "Aplikasi ini menggunakan dua model **Jaringan Syaraf Tiruan**: "
    "**Perceptron** dan **Kohonen Self-Organizing Map (SOM)**."
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**📌 Analisis**\n\nMasukkan data harian dan lihat hasil klasifikasi + clustering.")
with col2:
    st.info("**📊 Dataset**\n\nLihat dataset yang digunakan untuk melatih model.")
with col3:
    st.info("**📉 Perceptron & 🧩 SOM**\n\nLihat grafik training dan hasil clustering.")

st.divider()

# Ringkasan dataset
df, fitur, X, y, X_norm, nilai_min, nilai_max, perceptron, som = load_model()

st.subheader("📋 Ringkasan Data")
c1, c2, c3 = st.columns(3)
c1.metric("Total Data", len(df))
c2.metric("Produktif", int((y == 1).sum()))
c3.metric("Tidak Produktif", int((y == 0).sum()))

st.write("👈 Pilih halaman dari **sidebar kiri** untuk mulai.")