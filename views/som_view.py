import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def render(repo, som):
    """Render halaman clustering Kohonen SOM."""
    st.header("🧩 Hasil Clustering Kohonen SOM")

    clusters = som.predict(repo.X_norm)
    df_cluster = repo.df.copy()
    df_cluster["cluster"] = clusters
    df_cluster["nama_cluster"] = [som.get_cluster_name(c) for c in clusters]

    st.subheader("Tabel Hasil Clustering")
    st.dataframe(df_cluster, width="stretch")

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
    ax.set_title("Visualisasi Cluster SOM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.divider()

    st.subheader("Bobot Akhir SOM")
    bobot_som = pd.DataFrame(som.weights, columns=repo.fitur)
    bobot_som.insert(0, "Cluster", [som.get_cluster_name(i) for i in range(len(som.weights))])
    st.dataframe(bobot_som, width="stretch")
