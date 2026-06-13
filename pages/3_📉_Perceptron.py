import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import load_model

st.set_page_config(page_title="Training Perceptron", page_icon="📉", layout="wide")

st.header("📉 Hasil Training Perceptron")

df, fitur, X, y, X_norm, nilai_min, nilai_max, perceptron, som = load_model()

# Grafik error
st.subheader("Kurva Error per Epoch")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker="o", color="#4A90D9")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Error")
ax.set_title("Kurva Error Perceptron")
ax.grid(True)
st.pyplot(fig)

st.divider()

# Bobot akhir
col1, col2 = st.columns(2)
with col1:
    st.subheader("Bobot Akhir")
    bobot_df = pd.DataFrame({"Fitur": fitur, "Bobot": perceptron.weights})
    st.dataframe(bobot_df, use_container_width=True)

with col2:
    st.subheader("Bias Akhir")
    st.metric("Bias", round(perceptron.bias, 4))
