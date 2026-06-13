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

    st.divider()
    st.subheader("📖 Teori & Rumus Perhitungan Perceptron")

    st.markdown(f"""
    Model JST Perceptron adalah algoritma *Supervised Learning* yang digunakan untuk klasifikasi biner. Selama proses training, bobot ($w$) dan bias ($b$) diperbarui terus-menerus menggunakan rumus berikut:

    ##### 1. Persamaan Net Input (Kombinasi Linear)
    $$y_{{in}} = \\sum_{{i=1}}^n x_i w_i + b$$

    ##### 2. Fungsi Aktivasi (Undak Biner / Step Function)
    $$y = f(y_{{in}}) = \\begin{{cases}} 1 & \\text{{jika }} y_{{in}} \\ge 0 \\\\ 0 & \\text{{jika }} y_{{in}} < 0 \\end{{cases}}$$

    ##### 3. Rumus Pembaruan Bobot dan Bias (Delta Rule)
    Jika terdapat error ($e = y_{{\\text{{target}}}} - y_{{\\text{{prediksi}}}}$) antara target dan prediksi:
    $$\\Delta w_i = \\alpha \\times e \\times x_i$$
    $$\\Delta b = \\alpha \\times e$$

    Setelah delta dihitung, bobot dan bias baru diperbarui dengan:
    $$w_i \\leftarrow w_i + \\Delta w_i$$
    $$b \\leftarrow b + \\Delta b$$

    *Keterangan:*
    *   $x_i$: Nilai fitur ke-$i$ (Jam Tidur, Mood, Stres, Belajar, HP, Tugas)
    *   $w_i$: Bobot fitur ke-$i$ (Bobot akhir ditampilkan pada tabel di atas)
    *   $b$: Bias model (Bias akhir ditampilkan pada kolom di atas)
    *   $\\alpha$: *Learning Rate* (pada model ini diatur sebesar `{perceptron.learning_rate}`)
    *   $e$: Error (selisih target dengan prediksi)
    """)
