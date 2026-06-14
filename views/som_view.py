import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.styles import apply_custom_styles


def render(repo, som):
    """Render halaman clustering Kohonen (SOM)."""
    apply_custom_styles()
    st.header("🧩 Hasil Clustering Kohonen (SOM)")

    clusters = som.predict(repo.X_norm)
    df_cluster = repo.df.copy()
    df_cluster["cluster"] = clusters
    df_cluster["nama_cluster"] = [som.get_cluster_name(c) for c in clusters]

    st.subheader("📋 Tabel Hasil Clustering")
    st.dataframe(df_cluster, width="stretch")

    st.write("#### 🔍 Langkah Perhitungan Jarak Euclidean Per Baris Data")
    st.write("Pilih salah satu baris indeks data dari tabel di atas untuk melihat rincian perhitungan matematika jarak Euclidean ke setiap centroid cluster.")

    # Dropdown select box
    row_idx = st.selectbox(
        "Pilih Baris Data Latih:",
        options=range(len(repo.df)),
        format_func=lambda i: f"Baris {i+1} (Tidur: {repo.df.iloc[i]['jam_tidur']}, Mood: {repo.df.iloc[i]['mood']}, Stres: {repo.df.iloc[i]['stres']}, Belajar: {repo.df.iloc[i]['jam_belajar']}, HP: {repo.df.iloc[i]['jam_hp']}, Tugas: {repo.df.iloc[i]['jumlah_tugas']})"
    )

    # Tampilkan rincian data terpilih
    data_raw = repo.X[row_idx]
    data_norm = repo.X_norm[row_idx]
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("**Data Asli (Skala Riil):**")
        st.dataframe(pd.DataFrame([data_raw], columns=repo.fitur), width="stretch")
    with col_d2:
        st.write("**Data Ternormalisasi (Skala [0, 1]):**")
        st.dataframe(pd.DataFrame([data_norm], columns=repo.fitur), width="stretch")
    distances = np.linalg.norm(som.weights - data_norm, axis=1)
    winner_cluster_id = np.argmin(distances)

    for i in range(len(som.weights)):
        cluster_name = som.get_cluster_name(i)
        w_c = som.weights[i]
        # Hitung selisih kuadrat per fitur
        selisih_kuadrat = [(data_norm[j] - w_c[j])**2 for j in range(6)]
        langkah_sum = " + ".join([f"({data_norm[j]:.4f} - {w_c[j]:.4f})^2" for j in range(6)])
        langkah_num = " + ".join([f"{s:.4f}" for s in selisih_kuadrat])
        total_sum = sum(selisih_kuadrat)
        is_winner = "🏆 (Pemenang - Terdekat)" if i == winner_cluster_id else ""
        with st.expander(f"🔍 Jarak ke Centroid {cluster_name} {is_winner}"):
            st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{langkah_sum}}}")
            st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{langkah_num}}}")
            st.latex(rf"d_{{\text{{{cluster_name}}}}} = \sqrt{{{total_sum:.4f}}} = {np.sqrt(total_sum):.4f}")
    st.success(f"**Kesimpulan:** Baris {row_idx + 1} memiliki jarak terkecil ke centroid **{som.get_cluster_name(winner_cluster_id)}** ({np.min(distances):.4f}), sehingga dimasukkan ke klaster **{som.get_cluster_name(winner_cluster_id)}**.")

    st.divider()

    st.subheader("Visualisasi Cluster (Mood vs Stres)")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Modern premium dark console style
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    colors = ["#F87171", "#60A5FA", "#34D399"] # Glowing coral red, bright blue, emerald green

    # 1. Plot seluruh data latih
    for idx, cluster_id in enumerate(np.unique(clusters)):
        data_cluster = df_cluster[df_cluster["cluster"] == cluster_id]
        ax.scatter(
            data_cluster["mood"],
            data_cluster["stres"],
            label=som.get_cluster_name(cluster_id),
            color=colors[idx % len(colors)],
            alpha=0.8,
            edgecolors="#1E293B",
            linewidths=0.7,
            s=95,
            zorder=3
        )

    # 2. Plot Centroid Kohonen (Pusat Cluster / Bobot Model)
    for idx, w in enumerate(som.weights):
        # Denormalisasi bobot mood (index 1) dan stres (index 2) ke skala asli
        mood_c = w[1] * (repo.nilai_max[1] - repo.nilai_min[1]) + repo.nilai_min[1]
        stres_c = w[2] * (repo.nilai_max[2] - repo.nilai_min[2]) + repo.nilai_min[2]
        ax.scatter(
            mood_c,
            stres_c,
            color=colors[idx % len(colors)],
            edgecolors="#F8FAFC",
            s=220,
            marker="X",
            linewidths=1.5,
            label=f"Pusat Centroid: {som.get_cluster_name(idx)}",
            zorder=4
        )

    # 3. Plot data baris terpilih dari dropdown
    selected_mood = repo.df.iloc[row_idx]["mood"]
    selected_stres = repo.df.iloc[row_idx]["stres"]
    ax.scatter(
        selected_mood,
        selected_stres,
        color="#FBBF24", # Gold-yellow glow
        edgecolors="black",
        s=260,
        marker="*",
        linewidths=1.2,
        label=f"Data Baris Terpilih ({row_idx + 1})",
        zorder=5
    )

    ax.set_xlabel("Mood Pagi", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
    ax.set_ylabel("Tingkat Stres", fontsize=10, fontweight='semibold', color='#94A3B8', labelpad=8)
    ax.set_title("Visualisasi Pembagian Cluster Kohonen (SOM) & Centroid", fontsize=12, fontweight='bold', color='#F8FAFC', pad=15)
    ax.legend(frameon=True, facecolor='#1E293B', edgecolor='#334155', fontsize=9)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.15, color="#94A3B8")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')

    st.pyplot(fig)

    st.divider()

    st.subheader("⚖️ Bobot Akhir Kohonen (SOM)")
    bobot_som = pd.DataFrame(som.weights, columns=repo.fitur)
    bobot_som.insert(0, "Cluster", [som.get_cluster_name(i) for i in range(len(som.weights))])
    st.dataframe(bobot_som, width="stretch")

    st.divider()
    st.subheader("📖 Teori & Rumus Perhitungan Kohonen (SOM)")

    st.markdown(f"""
    Self-Organizing Map (SOM) atau Jaringan Kohonen adalah algoritma *Unsupervised Learning* yang mengelompokkan data berdasarkan kemiripan karakteristik tanpa memerlukan label target. Model ini dilatih melalui dua langkah utama:

    ##### 1. Pemilihan Best Matching Unit (BMU / Neuron Pemenang)
    Untuk setiap data input ($x$), model mencari neuron/klaster dengan bobot ($w_j$) terdekat menggunakan rumus **Jarak Euclidean**:
    $$d_j = \\sqrt{{\\sum_{{i=1}}^n (x_i - w_{{ji}})^2}}$$
    $$c = \\arg\\min_j (d_j)$$
    Dimana $c$ adalah indeks neuron pemenang (BMU) yang memiliki jarak Euclidean terkecil.

    ##### 2. Pembaruan Bobot (Weight Update)
    Hanya bobot neuron pemenang ($c$) yang diperbarui mendekati data input ($x$) menggunakan rumus:
    $$w_{{c}}(t+1) = w_{{c}}(t) + \\alpha(t) \\cdot (x - w_{{c}}(t))$$

    *Keterangan:*
    *   $x$: Vektor data input (dalam bentuk ternormalisasi)
    *   $w_j$: Vektor bobot untuk klaster ke-$j$ (Bobot akhir ditampilkan pada tabel di atas)
    *   $\\alpha(t)$: *Learning Rate* pada epoch $t$ (pada model ini diatur awal sebesar `{som.learning_rate}`)
    *   $t$: Epoch saat ini (total epoch: `{som.epoch}`)
    """)

