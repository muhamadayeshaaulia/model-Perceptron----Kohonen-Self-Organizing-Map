import streamlit as st
import numpy as np
import pandas as pd
from utils.helper import normalisasi_data_baru, get_saran, cek_konsistensi
import matplotlib.pyplot as plt


def render(repo, perceptron, som):
    """Render halaman input & analisis produktivitas mahasiswa."""
    st.header("📌 Input Data Harian Mahasiswa")
    st.write("Masukkan data aktivitas harian kamu untuk dianalisis oleh Perceptron dan SOM.")

    col1, col2 = st.columns(2)
    with col1:
        jam_tidur    = st.number_input("🛌 Jam Tidur (3–8 jam)", value=7, step=1)
        if jam_tidur < repo.nilai_min[0] or jam_tidur > repo.nilai_max[0]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[0])} - {int(repo.nilai_max[0])})</p>", unsafe_allow_html=True)
        mood         = st.number_input("😊 Mood Pagi (3–9)", value=7, step=1)
        if mood < repo.nilai_min[1] or mood > repo.nilai_max[1]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[1])} - {int(repo.nilai_max[1])})</p>", unsafe_allow_html=True)
        stres        = st.number_input("😤 Tingkat Stres (1–10)", value=4, step=1)
        if stres < repo.nilai_min[2] or stres > repo.nilai_max[2]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[2])} - {int(repo.nilai_max[2])})</p>", unsafe_allow_html=True)

    with col2:
        jam_belajar  = st.number_input("📚 Jam Belajar (1–6 jam)", value=4, step=1)
        if jam_belajar < repo.nilai_min[3] or jam_belajar > repo.nilai_max[3]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[3])} - {int(repo.nilai_max[3])})</p>", unsafe_allow_html=True)
        jam_hp       = st.number_input("📱 Jam Main HP (1–9 jam)", value=3, step=1)
        if jam_hp < repo.nilai_min[4] or jam_hp > repo.nilai_max[4]:
            st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[4])} - {int(repo.nilai_max[4])})</p>", unsafe_allow_html=True)
        jumlah_tugas = st.number_input("📝 Jumlah Tugas (1–8)", value=2, step=1)
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

    pesan_inconsistent = cek_konsistensi(jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas)

    if any_out_of_bounds:
        st.error("❌ Analisis tidak dapat dilakukan karena ada input di luar batas data latih. Silakan sesuaikan nilai input Anda.")
    elif pesan_inconsistent:
        st.error("❌ **Data Input Tidak Konsisten!** Silakan sesuaikan kembali input Anda agar logis:")
        for p in pesan_inconsistent:
            st.markdown(f"- {p}")

    data_baru = np.array(
        [[jam_tidur, mood, stres, jam_belajar, jam_hp, jumlah_tugas]],
        dtype=float
    )

    is_disabled = any_out_of_bounds or len(pesan_inconsistent) > 0

    if st.button("🔍 Analisis Sekarang", use_container_width=True, disabled=is_disabled):
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

        st.divider()
        st.subheader("🔍 Detail Matematis & Validasi Model")

        tab1, tab2, tab3 = st.tabs(["📊 Diagram Posisi Data", "🧠 Bobot & Bias Perceptron", "🧩 Validasi Cluster SOM"])

        with tab1:
            st.write("##### Posisi Data Baru pada Pembagian Cluster (Mood vs Stres)")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#E74C3C", "#3498DB", "#2ECC71"]
            
            # Ambil cluster dari data latih
            clusters_train = som.predict(repo.X_norm)
            df_cluster = repo.df.copy()
            df_cluster["cluster"] = clusters_train
            
            for idx, cluster_id in enumerate(np.unique(clusters_train)):
                data_cluster = df_cluster[df_cluster["cluster"] == cluster_id]
                ax.scatter(
                    data_cluster["mood"],
                    data_cluster["stres"],
                    label=som.get_cluster_name(cluster_id),
                    color=colors[idx % len(colors)],
                    alpha=0.6,
                    edgecolors="white",
                    s=60
                )
            
            # Plot data baru
            ax.scatter(
                mood,
                stres,
                color="#F1C40F",
                edgecolors="black",
                s=250,
                marker="*",
                label="Data Baru Anda",
                zorder=5
            )
            
            ax.set_xlabel("Mood Pagi")
            ax.set_ylabel("Tingkat Stres")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab2:
            st.write("##### Perhitungan Net Input Perceptron")
            bobot = perceptron.weights
            bias = perceptron.bias
            kontribusi = data_baru_clipped[0] * bobot
            total_net_input = np.sum(kontribusi) + bias
            
            df_perceptron = pd.DataFrame({
                "Fitur": repo.fitur,
                "Nilai Input (Clipped)": data_baru_clipped[0],
                "Bobot Model": bobot,
                "Kontribusi (Nilai x Bobot)": kontribusi
            })
            st.dataframe(df_perceptron, width="stretch")
            st.markdown("**Persamaan Linear:**")
            st.latex(rf"\text{{Net Input}} = \sum (\text{{Nilai}} \times \text{{Bobot}}) + \text{{Bias}}")
            st.latex(rf"\text{{Net Input}} = {total_net_input:.4f}")
            st.markdown(f"**Bias Model**: `{bias:.4f}`")
            status_aktivasi = ">= 0" if total_net_input >= 0 else "< 0"
            st.markdown(f"**Hasil Aktivasi**: `Net Input {status_aktivasi} \rightarrow` **{hasil_perceptron}**")

        with tab3:
            st.write("##### Jarak Euclidean ke Centroid SOM")
            distances = np.linalg.norm(som.weights - data_baru_norm, axis=1)
            
            df_som_val = pd.DataFrame({
                "Nama Cluster": [som.get_cluster_name(i) for i in range(len(som.weights))],
                "Jarak Euclidean ke Centroid": distances,
                "Status": ["Pemenang (Terdekat) 🏆" if i == cluster else "Bukan Pemenang" for i in range(len(som.weights))]
            })
            st.dataframe(df_som_val, width="stretch")
            st.write("*Catatan: SOM mengklasifikasikan input ke cluster dengan jarak Euclidean terkecil.*")

            st.write("##### Bobot Centroid (Pusat Koordinat) Masing-Masing Cluster SOM")
            bobot_som = pd.DataFrame(som.weights, columns=repo.fitur)
            bobot_som.insert(0, "Cluster", [som.get_cluster_name(i) for i in range(len(som.weights))])
            st.dataframe(bobot_som, width="stretch")
