import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render(repo, perceptron):
    """Render halaman hasil training Perceptron."""
    st.header("📉 Hasil Training Perceptron")

    st.subheader("Kurva Error per Epoch")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(1, len(perceptron.errors) + 1),
        perceptron.errors,
        marker="o",
        color="#4A90D9"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Error")
    ax.set_title("Kurva Error Perceptron")
    ax.grid(True)
    st.pyplot(fig)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bobot Akhir")
        bobot_df = pd.DataFrame({"Fitur": repo.fitur, "Bobot": perceptron.weights})
        st.dataframe(bobot_df, width="stretch")
    with col2:
        st.subheader("Bias Akhir")
        st.metric("Bias", round(float(perceptron.bias), 4))
