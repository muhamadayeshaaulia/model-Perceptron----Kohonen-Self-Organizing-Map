import streamlit as st
import numpy as np
import pandas as pd
from utils.helper import normalisasi_data_baru, get_saran, cek_konsistensi
import matplotlib.pyplot as plt


def render(repo, perceptron, som):
    """Render halaman input & analisis produktivitas mahasiswa."""
    st.header("📌 Input Data Harian Mahasiswa")
    st.write("Masukkan data aktivitas harian kamu untuk dianalisis oleh Perceptron dan Kohonen (SOM).")

    col1, col2 = st.columns(2)
    with col1:
        jam_tidur    = st.number_input("🛌 Jam Tidur (3–8 jam)", value=None, step=1)
        if jam_tidur is not None:
            if jam_tidur < repo.nilai_min[0] or jam_tidur > repo.nilai_max[0]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[0])} - {int(repo.nilai_max[0])})</p>", unsafe_allow_html=True)
        mood         = st.number_input("😊 Mood Pagi (3–9)", value=None, step=1)
        if mood is not None:
            if mood < repo.nilai_min[1] or mood > repo.nilai_max[1]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[1])} - {int(repo.nilai_max[1])})</p>", unsafe_allow_html=True)
        stres        = st.number_input("😤 Tingkat Stres (1–10)", value=None, step=1)
        if stres is not None:
            if stres < repo.nilai_min[2] or stres > repo.nilai_max[2]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[2])} - {int(repo.nilai_max[2])})</p>", unsafe_allow_html=True)

    with col2:
        jam_belajar  = st.number_input("📚 Jam Belajar (1–6 jam)", value=None, step=1)
        if jam_belajar is not None:
            if jam_belajar < repo.nilai_min[3] or jam_belajar > repo.nilai_max[3]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[3])} - {int(repo.nilai_max[3])})</p>", unsafe_allow_html=True)
        jam_hp       = st.number_input("📱 Jam Main HP (1–9 jam)", value=None, step=1)
        if jam_hp is not None:
            if jam_hp < repo.nilai_min[4] or jam_hp > repo.nilai_max[4]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[4])} - {int(repo.nilai_max[4])})</p>", unsafe_allow_html=True)
        jumlah_tugas = st.number_input("📝 Jumlah Tugas (1–8)", value=None, step=1)
        if jumlah_tugas is not None:
            if jumlah_tugas < repo.nilai_min[5] or jumlah_tugas > repo.nilai_max[5]:
                st.markdown(f"<p style='color: #E65100; margin-top: -15px; font-size: 0.85rem;'>⚠️ Di luar batas data latih ({int(repo.nilai_min[5])} - {int(repo.nilai_max[5])})</p>", unsafe_allow_html=True)

    # Cek apakah ada inputan yang kosong
    ada_yang_kosong = (
        jam_tidur is None or 
        mood is None or 
        stres is None or 
        jam_belajar is None or 
        jam_hp is None or 
        jumlah_tugas is None
    )

    if ada_yang_kosong:
        st.info("💡 **Silakan isi semua data harian di atas untuk memulai analisis.**")
        any_out_of_bounds = False
        pesan_inconsistent = []
        is_disabled = True
        data_baru = np.zeros((1, 6))
    else:
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

        # Prediksi Kohonen (SOM)
        data_baru_norm = normalisasi_data_baru(data_baru_clipped, repo.nilai_min, repo.nilai_max)
        cluster = som.predict(data_baru_norm)[0]
        hasil_som = som.get_cluster_name(cluster)

        st.divider()
        st.subheader("✅ Hasil Analisis")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.metric("Hasil Perceptron", hasil_perceptron)
        with col_h2:
            st.metric("Hasil Kohonen (Cluster)", hasil_som)

        st.info(get_saran(hasil_perceptron, hasil_som))

        st.write("### 📋 Data yang Dianalisis")
        st.dataframe(pd.DataFrame(data_baru, columns=repo.fitur), width="stretch")

        st.divider()
        st.subheader("🔍 Detail Matematis & Validasi Model")

        tab1, tab2, tab3 = st.tabs(["📊 Diagram Kohonen (SOM)", "🧠 Bobot & Bias Perceptron", "🧩 Validasi Jarak Kohonen"])

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
            
            # Plot Centroid Kohonen (Pusat Cluster / Bobot Model)
            for idx, w in enumerate(som.weights):
                # Denormalisasi bobot mood (index 1) dan stres (index 2) ke skala asli
                mood_c = w[1] * (repo.nilai_max[1] - repo.nilai_min[1]) + repo.nilai_min[1]
                stres_c = w[2] * (repo.nilai_max[2] - repo.nilai_min[2]) + repo.nilai_min[2]
                ax.scatter(
                    mood_c,
                    stres_c,
                    color=colors[idx % len(colors)],
                    edgecolors="black",
                    s=180,
                    marker="X",
                    linewidths=1.5,
                    label=f"Pusat Cluster: {som.get_cluster_name(idx)}",
                    zorder=4
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
            st.markdown(f"**Bias Model**: `{bias:.4f}`")
            
            st.markdown("**Langkah Perhitungan Manual Perceptron:**")
            langkah_perkalian = " + ".join([f"({data_baru_clipped[0][i]:.1f} \\times {bobot[i]:.4f})" for i in range(len(bobot))])
            st.latex(rf"\text{{Net Input}} = {langkah_perkalian} + ({bias:.4f})")
            st.latex(rf"\text{{Net Input}} = {np.sum(kontribusi):.4f} + ({bias:.4f}) = {total_net_input:.4f}")

            status_aktivasi = ">= 0" if total_net_input >= 0 else "< 0"
            st.markdown(f"**Hasil Aktivasi**: `Net Input {status_aktivasi} ->` **{hasil_perceptron}**")

            st.write("##### Visualisasi Garis Keputusan (Decision Boundary) Perceptron")
            # Plot Mood vs Stres dengan pemisahan kelas Perceptron
            fig_p, ax_p = plt.subplots(figsize=(8, 4))
            
            # Scatter plot data asli berdasarkan label kelas
            colors_p = {0: "#E74C3C", 1: "#2ECC71"} # 0 = Tidak Produktif, 1 = Produktif
            for label_id in [0, 1]:
                data_label = repo.df[repo.df["label"] == label_id]
                ax_p.scatter(
                    data_label["mood"],
                    data_label["stres"],
                    label="Produktif (Data Latih)" if label_id == 1 else "Tidak Produktif (Data Latih)",
                    color=colors_p[label_id],
                    alpha=0.6,
                    edgecolors="white",
                    s=60
                )
            
            # Hitung decision boundary line: w_mood * mood + w_stres * stres + Const = 0
            # Const = bias + sum_{j != mood, stres} (w_j * x_j)
            w = perceptron.weights
            b = perceptron.bias
            x_val = data_baru_clipped[0]
            
            # index 0: tidur, 1: mood, 2: stres, 3: belajar, 4: hp, 5: tugas
            const_val = b + w[0]*x_val[0] + w[3]*x_val[3] + w[4]*x_val[4] + w[5]*x_val[5]
            
            # Ambil rentang mood dari data latih untuk menggambar garis
            mood_range = np.linspace(repo.nilai_min[1], repo.nilai_max[1], 100)
            
            if w[2] != 0:
                # stres = - (w_mood * mood + const) / w_stres
                stres_range = - (w[1] * mood_range + const_val) / w[2]
                
                # Batasi stres_range agar tidak melenceng terlalu jauh dari plot
                ax_p.plot(mood_range, stres_range, color="#34495E", linestyle="--", linewidth=2.5, label="Garis Pemisah Perceptron")
            elif w[1] != 0:
                # Vertical line: mood = - const / w_mood
                mood_c = - const_val / w[1]
                ax_p.axvline(x=mood_c, color="#34495E", linestyle="--", linewidth=2.5, label="Garis Pemisah Perceptron")
                
            # Plot data baru
            ax_p.scatter(
                mood,
                stres,
                color="#F1C40F",
                edgecolors="black",
                s=250,
                marker="*",
                label="Data Baru Anda",
                zorder=5
            )
            
            ax_p.set_xlabel("Mood Pagi")
            ax_p.set_ylabel("Tingkat Stres")
            ax_p.legend()
            ax_p.grid(True, alpha=0.3)
            ax_p.set_ylim(0, 11) # Rentang wajar tingkat stres
            st.pyplot(fig_p)
            st.write("*Catatan: Garis pemisah di atas digambar dengan mengasumsikan 4 fitur lainnya (Jam Tidur, Jam Belajar, Jam HP, Jumlah Tugas) bernilai konstan sesuai input baru Anda. Karena data latih lainnya memiliki nilai fitur pendukung yang bervariasi, beberapa titik data latih mungkin terlihat berada di sisi garis yang tidak sesuai dengan warna kelasnya.*")

        with tab3:
            st.write("##### Jarak Euclidean ke Centroid Kohonen (SOM)")
            distances = np.linalg.norm(som.weights - data_baru_norm, axis=1)
            
            df_som_val = pd.DataFrame({
                "Nama Cluster": [som.get_cluster_name(i) for i in range(len(som.weights))],
                "Jarak Euclidean ke Centroid": distances,
                "Status": ["Pemenang (Terdekat) 🏆" if i == cluster else "Bukan Pemenang" for i in range(len(som.weights))]
            })
            st.dataframe(df_som_val, width="stretch")
            st.write("*Catatan: Kohonen (SOM) mengklasifikasikan input ke cluster dengan jarak Euclidean terkecil.*")

            st.write("##### Langkah Perhitungan Manual Jarak Euclidean (Kohonen SOM)")
            st.markdown("Rumus: $d = \\sqrt{\\sum_{j=1}^6 (x_j - w_j)^2}$")
            
            # data baru ter-normalisasi
            x_norm = data_baru_norm[0]
            
            for i in range(len(som.weights)):
                cluster_name = som.get_cluster_name(i)
                w_c = som.weights[i]
                
                # Hitung selisih kuadrat per fitur
                selisih_kuadrat = [(x_norm[j] - w_c[j])**2 for j in range(6)]
                langkah_sum = " + ".join([f"({x_norm[j]:.4f} - {w_c[j]:.4f})^2" for j in range(6)])
                langkah_num = " + ".join([f"{s:.4f}" for s in selisih_kuadrat])
                total_sum = sum(selisih_kuadrat)
                
                with st.expander(f"🔍 Langkah Perhitungan ke Cluster {cluster_name}"):
                    st.markdown(f"**Jarak ke Centroid {cluster_name}:**")
                    st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{langkah_sum}}}")
                    st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{langkah_num}}}")
                    st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{total_sum:.4f}}} = {np.sqrt(total_sum):.4f}")

            st.write("##### Bobot Centroid (Pusat Koordinat) Masing-Masing Cluster Kohonen (SOM)")
            bobot_som = pd.DataFrame(som.weights, columns=repo.fitur)
            bobot_som.insert(0, "Cluster", [som.get_cluster_name(i) for i in range(len(som.weights))])
            st.dataframe(bobot_som, width="stretch")
