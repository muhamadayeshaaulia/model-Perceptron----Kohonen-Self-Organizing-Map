import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import load_model

st.set_page_config(page_title="Clustering SOM", page_icon="🧩", layout="wide")

st.header("🧩 Hasil Clustering Kohonen SOM")

df, fitur, X, y, X_norm, nilai_min, nilai_max, perceptron, som = load_model()

clusters = som.predict(X_norm)
df_cluster = df.copy()
df_cluster["cluster"] = clusters
df_cluster["nama_cluster"] = [som.get_cluster_name(c) for c in clusters]

st.subheader("Tabel Hasil Clustering")
st.dataframe(df_cluster, use_container_width=True)

st.divider()

# Visualisasi scatter
st.subheader("Visualisasi Cluster (Mood vs Stres)")
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#E74C3C", "#3498DB", "#2ECC71"]

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

# Bobot SOM
st.subheader("Bobot Akhir SOM")
bobot_som = pd.DataFrame(som.weights, columns=fitur)
bobot_som.insert(0, "Cluster", [som.get_cluster_name(i) for i in range(len(som.weights))])
st.dataframe(bobot_som, use_container_width=True)
