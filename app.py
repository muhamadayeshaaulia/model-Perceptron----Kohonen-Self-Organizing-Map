import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from som import KohonenSOM
from helper import normalisasi, normalisasi_data_baru, get_saran





# =========================
# Konfigurasi halaman
# =========================

st.set_page_config(
    page_title="Deteksi Produktivitas Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Aplikasi Deteksi Produktivitas dan Burnout Mahasiswa")
st.write(
    "Aplikasi ini menggunakan dua model Jaringan Syaraf Tiruan, yaitu "
    "**Perceptron** dan **Kohonen Self-Organizing Map (SOM)**."
)


# =========================
# Load dataset
# =========================

df = pd.read_csv("dataset.csv")

fitur = ["jam_tidur", "mood", "stres", "jam_belajar", "jam_hp", "jumlah_tugas"]

X = df[fitur].values
y = df["label"].values

X_norm, nilai_min, nilai_max = normalisasi(X)


# =========================
# Training model
# =========================

perceptron = Perceptron(learning_rate=0.1, epoch=50)
perceptron.fit(X, y)

som = KohonenSOM(jumlah_cluster=3, learning_rate=0.5, epoch=100)
som.fit(X_norm)


# =========================
# Sidebar
# =========================

menu = st.sidebar.radio(
    "Menu",
    [
        "Analisis Mahasiswa",
        "Dataset",
        "Training Perceptron",
        "Clustering SOM",
        "Tentang Aplikasi"
    ]
)


# =========================
# Menu Analisis
# =========================

if menu == "Analisis Mahasiswa":
    st.header("📌 Input Data Harian Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        jam_tidur = st.slider("Jam Tidur", 0, 12, 7)
        mood = st.slider("Mood Pagi", 1, 10, 7)
        stres = st.slider("Tingkat Stres", 1, 10, 4)

    with col2:
        jam_belajar = st.slider("Jam Belajar", 0, 12, 4)
        jam_hp = st.slider("Jam Main HP", 0, 12, 3)
        jumlah_tugas = st.slider("Jumlah Tugas", 0, 10, 2)

    data_baru = np.array([[jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]], dtype=float)

    if st.button("Analisis Sekarang"):
        hasil_p = perceptron.predict(data_baru)[0]
        hasil_perceptron = "Produktif" if hasil_p == 1 else "Tidak Produktif"

        data_baru_norm = normalisasi_data_baru(data_baru, nilai_min, nilai_max)
        cluster = som.predict(data_baru_norm)[0]
        hasil_som = som.get_cluster_name(cluster)

        st.subheader("✅ Hasil Analisis")

        col_hasil1, col_hasil2 = st.columns(2)

        with col_hasil1:
            st.metric("Hasil Perceptron", hasil_perceptron)

        with col_hasil2:
            st.metric("Hasil SOM", hasil_som)

        st.info(get_saran(hasil_perceptron, hasil_som))

        st.write("### Data yang Dianalisis")
        data_tampil = pd.DataFrame(data_baru, columns=fitur)
        st.dataframe(data_tampil)


# =========================
# Menu Dataset
# =========================

elif menu == "Dataset":
    st.header("📊 Dataset Mahasiswa")
    st.write("Dataset ini digunakan sebagai data latih untuk model Perceptron dan SOM.")
    st.dataframe(df)

    st.write("Keterangan label:")
    st.write("- **1** = Produktif")
    st.write("- **0** = Tidak Produktif")


# =========================
# Menu Training Perceptron
# =========================

elif menu == "Training Perceptron":
    st.header("📉 Grafik Error Perceptron")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Error")
    ax.set_title("Kurva Error Perceptron")
    ax.grid(True)

    st.pyplot(fig)

    st.write("### Bobot Akhir Perceptron")
    bobot_df = pd.DataFrame({
        "Fitur": fitur,
        "Bobot": perceptron.weights
    })
    st.dataframe(bobot_df)

    st.write("Bias akhir:", perceptron.bias)


# =========================
# Menu Clustering SOM
# =========================

elif menu == "Clustering SOM":
    st.header("🧩 Hasil Clustering Kohonen SOM")

    clusters = som.predict(X_norm)
    df_cluster = df.copy()
    df_cluster["cluster"] = clusters
    df_cluster["nama_cluster"] = [som.get_cluster_name(c) for c in clusters]

    st.dataframe(df_cluster)

    st.write("### Visualisasi Cluster")
    st.write("Visualisasi menggunakan dua fitur: mood dan tingkat stres.")

    fig, ax = plt.subplots()

    for cluster_id in np.unique(clusters):
        data_cluster = df_cluster[df_cluster["cluster"] == cluster_id]
        ax.scatter(
            data_cluster["mood"],
            data_cluster["stres"],
            label=som.get_cluster_name(cluster_id)
        )

    ax.set_xlabel("Mood")
    ax.set_ylabel("Stres")
    ax.set_title("Visualisasi Cluster SOM")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.write("### Bobot Akhir SOM")
    bobot_som = pd.DataFrame(som.weights, columns=fitur)
    bobot_som["nama_cluster"] = [som.get_cluster_name(i) for i in range(len(som.weights))]
    st.dataframe(bobot_som)


# =========================
# Menu Tentang Aplikasi
# =========================

elif menu == "Tentang Aplikasi":
    st.header("ℹ️ Tentang Aplikasi")

    st.write("""
    Aplikasi ini dibuat untuk proyek UAS mata kuliah Jaringan Syaraf Tiruan.
    Tujuannya adalah membantu mahasiswa menganalisis kondisi produktivitas dan potensi burnout
    berdasarkan pola harian.
    """)

    st.write("### Model yang Digunakan")

    st.write("""
    **1. Perceptron**

    Perceptron digunakan untuk melakukan klasifikasi sederhana, yaitu menentukan apakah kondisi
    mahasiswa termasuk Produktif atau Tidak Produktif.
    """)

    st.write("""
    **2. Kohonen Self-Organizing Map (SOM)**

    Kohonen SOM digunakan untuk melakukan clustering atau pengelompokan data mahasiswa
    menjadi beberapa kategori gaya hidup, yaitu Produktif, Santai, dan Burnout.
    """)

    st.write("### Input Sistem")
    st.write("""
    - Jam tidur
    - Mood pagi
    - Tingkat stres
    - Jam belajar
    - Jam main HP
    - Jumlah tugas
    """)

    st.write("### Output Sistem")
    st.write("""
    - Hasil klasifikasi Perceptron
    - Hasil clustering SOM
    - Saran aktivitas harian
    """)