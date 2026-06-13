import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def render(repo, som):
    """Render halaman clustering Kohonen (SOM)."""
    st.header("🧩 Hasil Clustering Kohonen (SOM)")

    clusters = som.predict(repo.X_norm)
    df_cluster = repo.df.copy()
    df_cluster["cluster"] = clusters
    df_cluster["nama_cluster"] = [som.get_cluster_name(c) for c in clusters]

    st.subheader("Tabel Hasil Clustering")
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
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, cluster_id in enumerate(np.unique(clusters)):
        data_cluster = df_cluster[df_cluster["cluster"] == cluster_id]
        ax.scatter(
            data_cluster["mood"],
            data_cluster["stres"],
            label=som.get_cluster_name(cluster_id),
            color=colors[idx % len(colors)],
            alpha=0.8,
            edgecolors="white",
            s=80
        )

    ax.set_xlabel("Mood")
    ax.set_ylabel("Stres")
    ax.set_title("Visualisasi Cluster Kohonen (SOM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.divider()

    st.subheader("Bobot Akhir Kohonen (SOM)")
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
