import streamlit as st
import numpy as np
import pandas as pd
from utils.helper import normalisasi_data_baru, get_saran, cek_konsistensi
from utils.styles import apply_custom_styles
import matplotlib.pyplot as plt


def render(repo, perceptron, som):
    """Render halaman input & analisis produktivitas mahasiswa."""
    apply_custom_styles()
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
        # Cast agar type checker (Pyrefly) tahu nilai ini tidak None saat analisis berjalan
        mood_val = float(mood) if mood is not None else 0.0
        stres_val = float(stres) if stres is not None else 0.0

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

        # ── 2 Tab: Perceptron | Kohonen ──────────────────────────────────
        tab_perceptron, tab_kohonen = st.tabs(["🧠 Bobot & Bias Perceptron", "🧩 Validasi Jarak Kohonen"])

        # ─────────────────────────── TAB PERCEPTRON ──────────────────────
        with tab_perceptron:
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
            st.markdown(
                f"""<div style="background: rgba(79,70,229,0.08); border: 1px solid rgba(79,70,229,0.25); 
                border-radius: 10px; padding: 0.75rem 1rem; margin: 0.5rem 0;">
                <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; 
                letter-spacing: 0.05em; opacity: 0.7;">📌 Nilai Bias Model (tetap setelah training)</span><br/>
                <span style="font-size: 1.2rem; font-weight: 800; color: #818CF8;">b = {bias:.4f}</span>
                <span style="font-size: 0.82rem; opacity: 0.65; margin-left: 8px;">
                (Ini adalah hasil training — bukan hasil prediksi)</span>
                </div>""",
                unsafe_allow_html=True
            )

            st.markdown("**Langkah Perhitungan Manual Perceptron:**")
            langkah_perkalian = " + ".join([f"({data_baru_clipped[0][i]:.1f} \\times {bobot[i]:.4f})" for i in range(len(bobot))])
            st.latex(rf"\text{{Net Input}} = {langkah_perkalian} + ({bias:.4f})")
            st.latex(rf"\text{{Net Input}} = {np.sum(kontribusi):.4f} + ({bias:.4f}) = {total_net_input:.4f}")

            # Hasil aktivasi dengan badge warna yang jelas
            status_aktivasi = ">= 0" if total_net_input >= 0 else "< 0"
            warna_aktivasi = "#10B981" if total_net_input >= 0 else "#EF4444"
            emoji_aktivasi = "✅" if total_net_input >= 0 else "❌"
            nilai_y = 1 if total_net_input >= 0 else 0
            st.markdown(
                f"""<div style="background: rgba({('16,185,129' if total_net_input >= 0 else '239,68,68')},0.1); 
                border: 1.5px solid {warna_aktivasi}; border-radius: 10px; 
                padding: 0.85rem 1rem; margin: 0.75rem 0;">
                <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; 
                letter-spacing: 0.05em; color: {warna_aktivasi};">⚡ Hasil Fungsi Aktivasi Step (y = f(Net Input))</span><br/>
                <span style="font-size: 1rem; font-weight: 600; opacity: 0.85;">
                Net Input = <strong>{total_net_input:.4f}</strong> {status_aktivasi}
                </span><br/>
                <span style="font-size: 1.25rem; font-weight: 800; color: {warna_aktivasi};">
                {emoji_aktivasi} y = {nilai_y} → <strong>{hasil_perceptron}</strong>
                </span>
                </div>""",
                unsafe_allow_html=True
            )

            st.write("##### Visualisasi Garis Keputusan (Decision Boundary) Perceptron")
            fig_p, ax_p = plt.subplots(figsize=(8.5, 4.5))
            fig_p.patch.set_facecolor('#0F172A')
            ax_p.set_facecolor('#1E293B')
            colors_p = {0: "#F87171", 1: "#34D399"}
            for label_id in [0, 1]:
                data_label = repo.df[repo.df["label"] == label_id]
                ax_p.scatter(
                    data_label["mood"], data_label["stres"],
                    label="Produktif (Data Latih)" if label_id == 1 else "Tidak Produktif (Data Latih)",
                    color=colors_p[label_id], alpha=0.8, edgecolors="#1E293B", linewidths=0.7, s=80, zorder=3
                )
            w = perceptron.weights
            b = perceptron.bias
            x_val = data_baru_clipped[0]
            const_val = b + w[0]*x_val[0] + w[3]*x_val[3] + w[4]*x_val[4] + w[5]*x_val[5]
            mood_range = np.linspace(repo.nilai_min[1], repo.nilai_max[1], 100)
            if w[2] != 0:
                stres_range = - (w[1] * mood_range + const_val) / w[2]
                ax_p.plot(mood_range, stres_range, color="#818CF8", linestyle="--", linewidth=2.5, label="Garis Pemisah Perceptron", zorder=4)
            elif w[1] != 0:
                mood_c = - const_val / w[1]
                ax_p.axvline(x=mood_c, color="#818CF8", linestyle="--", linewidth=2.5, label="Garis Pemisah Perceptron", zorder=4)
            ax_p.scatter(mood_val, stres_val, color="#FBBF24", edgecolors="black", s=260, marker="*", linewidths=1.2, label="Data Baru Anda", zorder=5)
            ax_p.set_xlabel("Mood Pagi", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
            ax_p.set_ylabel("Tingkat Stres", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
            ax_p.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=9)
            ax_p.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
            ax_p.spines['top'].set_visible(False)
            ax_p.spines['right'].set_visible(False)
            ax_p.spines['left'].set_color('#334155')
            ax_p.spines['bottom'].set_color('#334155')
            ax_p.tick_params(colors='#94A3B8', labelsize=9)
            ax_p.set_ylim(0, 11)
            st.pyplot(fig_p)
            st.write("*Catatan: Garis pemisah di atas digambar dengan mengasumsikan 4 fitur lainnya (Jam Tidur, Jam Belajar, Jam HP, Jumlah Tugas) bernilai konstan sesuai input baru Anda. Karena data latih lainnya memiliki nilai fitur pendukung yang bervariasi, beberapa titik data latih mungkin terlihat berada di sisi garis yang tidak sesuai dengan warna kelasnya.*")

        # ─────────────────────────── TAB KOHONEN ─────────────────────────
        with tab_kohonen:
            # Diagram Scatter Kohonen di bagian atas — visualisasi posisi data baru
            st.write("##### 📊 Posisi Data Baru pada Cluster Kohonen (Mood vs Stres)")
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            fig.patch.set_facecolor('#0F172A')
            ax.set_facecolor('#1E293B')
            colors = ["#F87171", "#60A5FA", "#34D399"]

            clusters_train = som.predict(repo.X_norm)
            df_cluster = repo.df.copy()
            df_cluster["cluster"] = clusters_train

            for idx, cluster_id in enumerate(np.unique(clusters_train)):
                data_cluster = df_cluster[df_cluster["cluster"] == cluster_id]
                ax.scatter(
                    data_cluster["mood"], data_cluster["stres"],
                    label=som.get_cluster_name(cluster_id),
                    color=colors[idx % len(colors)], alpha=0.8,
                    edgecolors="#1E293B", linewidths=0.7, s=80, zorder=3
                )

            for idx, ww in enumerate(som.weights):
                mood_c = ww[1] * (repo.nilai_max[1] - repo.nilai_min[1]) + repo.nilai_min[1]
                stres_c = ww[2] * (repo.nilai_max[2] - repo.nilai_min[2]) + repo.nilai_min[2]
                ax.scatter(
                    mood_c, stres_c,
                    color=colors[idx % len(colors)], edgecolors="#F8FAFC",
                    s=180, marker="X", linewidths=1.5,
                    label=f"Pusat Cluster: {som.get_cluster_name(idx)}", zorder=4
                )

            ax.scatter(
                mood_val, stres_val,
                color="#FBBF24", edgecolors="black",
                s=260, marker="*", linewidths=1.2,
                label="Data Baru Anda ⭐", zorder=5
            )

            ax.set_xlabel("Mood Pagi", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
            ax.set_ylabel("Tingkat Stres", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
            ax.set_title("Posisi Data Baru pada Cluster Kohonen (SOM)", fontsize=11, fontweight='bold', color='#F8FAFC', pad=12)
            ax.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')
            ax.tick_params(colors='#94A3B8', labelsize=9)
            st.pyplot(fig)

            st.divider()

            # Tabel validasi jarak Euclidean
            st.write("##### 📐 Jarak Euclidean ke Centroid Kohonen (SOM)")
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

            x_norm = data_baru_norm[0]
            for i in range(len(som.weights)):
                cluster_name = som.get_cluster_name(i)
                w_c = som.weights[i]
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
